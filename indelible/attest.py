"""AttestRunner — execute test suite and produce a behavioral Fingerprint."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional

from indelible.config import IndelibleConfig
from indelible.providers import get_provider
from indelible.types import Fingerprint, Signal

logger = logging.getLogger(__name__)


def _test_set_hash(test_inputs: List[str]) -> str:
    payload = json.dumps(test_inputs, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _api_key(model: str, provider_url: Optional[str] = None) -> Optional[str]:
    """Return the right env-var API key for the given model + provider URL.

    Routing priority:
    1. Anthropic models → ANTHROPIC_API_KEY
    2. Ollama models → no key
    3. provider_url host match → host-specific env var
       (so a user with both OPENAI_API_KEY and GROQ_API_KEY set hits the
       right one even when the OpenAI-compat fallback is used)
    4. fallback iteration over OPENAI / GROQ / TOGETHER
    """
    if model.startswith("claude-") or model.startswith("anthropic/"):
        return os.environ.get("ANTHROPIC_API_KEY")
    if model.startswith("ollama/"):
        return None

    # URL-host routing: pick the env var that actually matches the target.
    if provider_url:
        url_l = provider_url.lower()
        if "groq.com" in url_l:
            return os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if "together.xyz" in url_l or "together.ai" in url_l:
            return os.environ.get("TOGETHER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if "openai.com" in url_l or "azure.com" in url_l:
            return os.environ.get("OPENAI_API_KEY")

    # Last-resort fallback: first set var wins.
    for var in ("OPENAI_API_KEY", "GROQ_API_KEY", "TOGETHER_API_KEY"):
        val = os.environ.get(var)
        if val:
            return val
    return None


def attest(
    config: IndelibleConfig,
    test_inputs: List[str],
    model: str,
    *,
    sign_key: Optional[str] = None,
    sig_out_path: Optional[str] = None,
) -> Fingerprint:
    """Run test suite against provider and return a signed behavioral Fingerprint.

    Args:
        config: Agent configuration (system prompt, tools, provider URL).
        test_inputs: Non-empty list of representative user prompts.
        model: Model identifier (e.g. ``"claude-opus-4-7"``).
        sign_key: Path to Ed25519 PEM private key. When given, writes a
            signature file (default: ``<sign_key>.sig`` — see ``sig_out_path``).
        sig_out_path: Where to write the ``.sig`` (and companion ``.sig.pub``).
            Recommended: next to the saved fingerprint, e.g.
            ``"indelible.fingerprint.json.sig"`` — sigs that travel with the
            artefact survive ``git pull`` into CI. Defaults to ``sign_key + ".sig"``
            for backward compatibility.

    Returns:
        A :class:`~indelible.types.Fingerprint` capturing agent behavior.
    """
    if not isinstance(config, IndelibleConfig):
        raise ValueError(f"config must be IndelibleConfig, got {type(config)}")
    test_inputs = copy.deepcopy(test_inputs)
    if not test_inputs:
        raise ValueError("test_inputs must be non-empty")

    provider = get_provider(
        model, config.provider_url, _api_key(model, config.provider_url),
        temperature=getattr(config, "temperature", 0.0),
    )

    outputs: List[str] = []
    tools_called: List[List[str]] = []
    latencies: List[float] = []

    for prompt in test_inputs:
        out, tc, lat = provider.complete(system=config.system_prompt, user=prompt)
        outputs.append(out)
        tools_called.append(tc)
        latencies.append(lat)

    signals = _run_collectors(config, outputs, test_inputs, tools_called, latencies)

    fp = Fingerprint(
        schema_version="1",
        config_hash=config.canonical_hash(),
        model=model,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        maintainer=getattr(config, "maintainer", "") or "",
        signals=tuple(signals),
        test_set_hash=_test_set_hash(test_inputs),
    )

    if sign_key:
        _sign(fp, sign_key, sig_out_path=sig_out_path)

    return fp


def _run_collectors(
    config: IndelibleConfig,
    outputs: List[str],
    inputs: List[str],
    tools_called: List[List[str]],
    latencies: List[float],
) -> List[Signal]:
    from indelible.collectors import Collector
    from indelible.collectors.refusal import RefusalCollector
    from indelible.collectors.latency import LatencyCollector
    from indelible.collectors.vocab_entropy import VocabEntropyCollector
    from indelible.collectors.tool_distribution import ToolDistributionCollector
    from indelible.collectors.tool_schema_hash import ToolSchemaHashCollector
    from indelible.collectors.embedding_profile import EmbeddingProfileCollector
    from indelible.collectors.anchor_drift import AnchorDriftCollector

    lat_collector = LatencyCollector()
    lat_collector.set_latencies(latencies)
    schema_collector = ToolSchemaHashCollector(config.tools)
    refusal_collector = RefusalCollector(patterns=getattr(config, "refusal_patterns", None))

    collectors: List[Collector] = [
        refusal_collector,
        lat_collector,
        VocabEntropyCollector(),
        ToolDistributionCollector(),
        schema_collector,
        EmbeddingProfileCollector(),
        AnchorDriftCollector(),
    ]

    signals: List[Signal] = []
    for collector in collectors:
        try:
            sig = collector.collect(
                outputs=outputs,
                inputs=inputs,
                anchor_text=config.system_prompt,
                tools_called=tools_called,
            )
            signals.append(sig)
        except Exception as exc:
            logger.warning("Collector %s failed, skipping: %s", collector.name, exc)

    return signals


def _sign(fp: Fingerprint, key_path: str, *, sig_out_path: Optional[str] = None) -> None:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    payload = fp.canonical_bytes()
    with open(key_path, "rb") as fh:
        private_key = load_pem_private_key(fh.read(), password=None)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise ValueError(f"sign_key must be Ed25519, got {type(private_key).__name__}")
    signature = private_key.sign(payload)  # Ed25519: no padding/hash args

    sig_path = sig_out_path if sig_out_path else key_path + ".sig"
    with open(sig_path, "wb") as fh:
        fh.write(signature)

    # NOTE: we deliberately do NOT write a companion public key next to the
    # signature. A pub that travels with the sig is no trust anchor — an
    # attacker edits the fingerprint, re-signs with their own key, and ships
    # their own pub. Authenticity requires a *pinned* public key the verifier
    # already trusts (committed `indelible.pub` or passed via --pubkey).
    logger.info("Signature written to %s", sig_path)
