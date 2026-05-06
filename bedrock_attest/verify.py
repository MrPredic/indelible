"""Verifier — compare a fresh attestation against a saved Fingerprint."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

from bedrock_attest.attest import attest
from bedrock_attest.config import BedrockConfig
from bedrock_attest.types import Fingerprint, Signal, VerifyReport

_VERDICT = {"pass": 0, "warn": 1, "breach": 2}
_RANK = {v: k for k, v in _VERDICT.items()}


def _compare_signal(original: Signal, fresh: Signal) -> Tuple[str, str, str]:
    delta = abs(fresh.value - original.value)
    if delta <= original.tolerance * 0.5:
        verdict = "pass"
    elif delta <= original.tolerance:
        verdict = "warn"
    else:
        verdict = "breach"
    detail = f"Δ {delta:+.4f} (tol ±{original.tolerance})"
    return original.name, verdict, detail


def _worst(verdicts: List[str]) -> str:
    return _RANK[max(_VERDICT[v] for v in verdicts)] if verdicts else "pass"


def verify(
    fingerprint_path: str,
    config: BedrockConfig,
    model: str,
    test_inputs: List[str],
    *,
    sig_path: Optional[str] = None,
) -> VerifyReport:
    """Compare current agent behavior against a saved Fingerprint.

    Args:
        fingerprint_path: Path to ``bedrock.fingerprint.json``.
        config: Current agent configuration.
        model: Model identifier to attest against.
        test_inputs: Same prompts used during original attestation.
        sig_path: Optional path to ``.sig`` file for signature verification.

    Returns:
        :class:`~bedrock_attest.types.VerifyReport` with per-signal verdicts.
    """
    t0 = time.perf_counter()

    with open(fingerprint_path, "r", encoding="utf-8") as fh:
        original = Fingerprint.from_dict(json.load(fh))

    if sig_path:
        _verify_signature(original, sig_path)

    fresh = attest(config=config, test_inputs=test_inputs, model=model)

    fresh_by_name = {s.name: s for s in fresh.signals}
    per_signal: List[Tuple[str, str, str]] = []

    for sig in original.signals:
        if sig.name in fresh_by_name:
            per_signal.append(_compare_signal(sig, fresh_by_name[sig.name]))
        else:
            per_signal.append((sig.name, "warn", "signal not collected in fresh run"))

    overall = _worst([v for _, v, _ in per_signal])
    elapsed = time.perf_counter() - t0

    return VerifyReport(
        overall=overall,
        per_signal=tuple(per_signal),
        elapsed_s=elapsed,
    )


def _verify_signature(fp: Fingerprint, sig_path: str) -> None:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.serialization import load_pem_public_key

    payload = json.dumps(fp.to_dict(), sort_keys=True, ensure_ascii=False).encode()
    with open(sig_path, "rb") as fh:
        signature = fh.read()

    pub_path = Path(sig_path).with_suffix(".pub")
    if not pub_path.exists():
        raise ValueError(f"Public key not found at {pub_path}")
    with open(pub_path, "rb") as fh:
        public_key = load_pem_public_key(fh.read())

    try:
        public_key.verify(signature, payload)  # Ed25519: raises InvalidSignature if bad
    except InvalidSignature as exc:
        raise ValueError("Fingerprint signature is invalid") from exc
