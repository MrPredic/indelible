"""AttestRunner — execute test suite and produce a behavioral Fingerprint."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional

from bedrock_attest.config import BedrockConfig
from bedrock_attest.providers import get_provider
from bedrock_attest.types import Fingerprint, Signal

logger = logging.getLogger(__name__)


def _test_set_hash(test_inputs: List[str]) -> str:
    payload = json.dumps(test_inputs, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _api_key(model: str) -> Optional[str]:
    if model.startswith("claude-") or model.startswith("anthropic/"):
        return os.environ.get("ANTHROPIC_API_KEY")
    return os.environ.get("OPENAI_API_KEY")


def attest(
    config: BedrockConfig,
    test_inputs: List[str],
    model: str,
    *,
    sign_key: Optional[str] = None,
) -> Fingerprint:
    """Run test suite against provider and return a signed behavioral Fingerprint.

    Args:
        config: Agent configuration (system prompt, tools, provider URL).
        test_inputs: Non-empty list of representative user prompts.
        model: Model identifier (e.g. ``"claude-opus-4-7"``).
        sign_key: Path to Ed25519 PEM private key. When given, writes a
            ``<sign_key>.sig`` file containing the signature.

    Returns:
        A :class:`~bedrock_attest.types.Fingerprint` capturing agent behavior.
    """
    if not isinstance(config, BedrockConfig):
        raise ValueError(f"config must be BedrockConfig, got {type(config)}")
    test_inputs = copy.deepcopy(test_inputs)
    if not test_inputs:
        raise ValueError("test_inputs must be non-empty")

    provider = get_provider(model, config.provider_url, _api_key(model))

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
        maintainer="",
        signals=tuple(signals),
        test_set_hash=_test_set_hash(test_inputs),
    )

    if sign_key:
        _sign(fp, sign_key)

    return fp


def _run_collectors(
    config: BedrockConfig,
    outputs: List[str],
    inputs: List[str],
    tools_called: List[List[str]],
    latencies: List[float],
) -> List[Signal]:
    from bedrock_attest.collectors.refusal import RefusalCollector
    from bedrock_attest.collectors.latency import LatencyCollector
    from bedrock_attest.collectors.vocab_entropy import VocabEntropyCollector
    from bedrock_attest.collectors.tool_distribution import ToolDistributionCollector
    from bedrock_attest.collectors.tool_schema_hash import ToolSchemaHashCollector
    from bedrock_attest.collectors.embedding_profile import EmbeddingProfileCollector
    from bedrock_attest.collectors.anchor_drift import AnchorDriftCollector

    lat_collector = LatencyCollector()
    lat_collector.set_latencies(latencies)
    schema_collector = ToolSchemaHashCollector(config.tools)

    collectors = [
        RefusalCollector(),
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


def _sign(fp: Fingerprint, key_path: str) -> None:
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    payload = json.dumps(fp.to_dict(), sort_keys=True, ensure_ascii=False).encode()
    with open(key_path, "rb") as fh:
        private_key = load_pem_private_key(fh.read(), password=None)
    signature = private_key.sign(payload)  # Ed25519: no padding/hash args
    sig_path = key_path + ".sig"
    with open(sig_path, "wb") as fh:
        fh.write(signature)
    logger.info("Signature written to %s", sig_path)
