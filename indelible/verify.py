"""Verifier — compare a fresh attestation against a saved Fingerprint."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, cast

from indelible.attest import _test_set_hash, attest
from indelible.config import IndelibleConfig
from indelible.types import Fingerprint, Signal, VerifyReport

_VERDICT = {"pass": 0, "warn": 1, "breach": 2}
_RANK = {v: k for k, v in _VERDICT.items()}


def _compare_signal(original: Signal, fresh: Signal) -> Tuple[str, str, str]:
    # Exact-match path: signals carrying a `digest` (SHA-style) are compared
    # bytewise. Any difference is a breach — no tolerance band.
    if original.digest is not None or fresh.digest is not None:
        if original.digest == fresh.digest and original.digest is not None:
            return original.name, "pass", f"digest match ({original.digest[:12]}…)"
        o = (original.digest or "")[:12] or "—"
        f = (fresh.digest or "")[:12] or "—"
        return original.name, "breach", f"digest mismatch ({o}… → {f}…)"

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
    config: IndelibleConfig,
    model: str,
    test_inputs: List[str],
    *,
    sig_path: Optional[str] = None,
    pubkey_path: Optional[str] = None,
) -> VerifyReport:
    """Compare current agent behavior against a saved Fingerprint.

    Args:
        fingerprint_path: Path to ``indelible.fingerprint.json``.
        config: Current agent configuration.
        model: Model identifier to attest against.
        test_inputs: Same prompts used during original attestation.
        sig_path: Optional path to ``.sig`` file for signature verification.
        pubkey_path: Path to the *pinned* Ed25519 public key (the trust
            anchor). **Required whenever ``sig_path`` is given** — verifying a
            signature without a pinned key proves nothing (the signer could be
            anyone). Point this at your committed ``indelible.pub`` or a key you
            obtained out-of-band from the attesting party.

    Returns:
        :class:`~indelible.types.VerifyReport` with per-signal verdicts.
    """
    t0 = time.perf_counter()

    with open(fingerprint_path, "r", encoding="utf-8") as fh:
        original = Fingerprint.from_dict(json.load(fh))

    if sig_path:
        if not pubkey_path:
            raise ValueError(
                "pubkey_path is required to verify a signature: a signature "
                "without a pinned public key is not a trust anchor. Pass the "
                "committed indelible.pub (or a key obtained out-of-band)."
            )
        _verify_signature(original, sig_path, pubkey_path)

    # Context guards: a fingerprint is only meaningful for the same config + test set.
    # Mismatch is breach (not warn) — silently passing here would defeat attestation.
    # Short-circuit before the live re-attestation: there's no point burning
    # provider tokens for a verdict that's already breach.
    per_signal: List[Tuple[str, str, str]] = []
    if original.config_hash != config.canonical_hash():
        per_signal.append(("config_hash", "breach",
                           "config differs from the one used at attest time"))
    fresh_test_hash = _test_set_hash(test_inputs)
    if original.test_set_hash != fresh_test_hash:
        per_signal.append(("test_set_hash", "breach",
                           "test prompts differ from the ones used at attest time"))

    if per_signal:
        elapsed = time.perf_counter() - t0
        return VerifyReport(
            overall=cast(Literal["pass", "warn", "breach"], "breach"),
            per_signal=tuple(per_signal),
            elapsed_s=elapsed,
        )

    fresh = attest(config=config, test_inputs=test_inputs, model=model)

    fresh_by_name = {s.name: s for s in fresh.signals}
    original_names = {s.name for s in original.signals}

    for sig in original.signals:
        if sig.name in fresh_by_name:
            per_signal.append(_compare_signal(sig, fresh_by_name[sig.name]))
        else:
            per_signal.append((sig.name, "warn", "signal not collected in fresh run"))

    # Surface signals present in the fresh run but missing from the original —
    # otherwise schema drift (new collector added between attest and verify)
    # is silently dropped.
    for name, sig in fresh_by_name.items():
        if name not in original_names:
            per_signal.append((name, "warn",
                               "signal present in fresh run but missing from saved fingerprint"))

    overall = cast(Literal["pass", "warn", "breach"], _worst([v for _, v, _ in per_signal]))
    elapsed = time.perf_counter() - t0

    return VerifyReport(
        overall=overall,
        per_signal=tuple(per_signal),
        elapsed_s=elapsed,
    )


def _verify_signature(fp: Fingerprint, sig_path: str, pubkey_path: str) -> None:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.hazmat.primitives.serialization import load_pem_public_key

    payload = fp.canonical_bytes()
    with open(sig_path, "rb") as fh:
        signature = fh.read()

    # Trust anchor: the *pinned* public key the caller already trusts, NOT a
    # pub that rode along next to the sig. Changing this key is a visible,
    # reviewable event (git diff of indelible.pub / an out-of-band re-pin).
    pub_path = Path(pubkey_path)
    if not pub_path.exists():
        raise ValueError(f"Pinned public key not found at {pub_path}")
    with open(pub_path, "rb") as fh:
        public_key = load_pem_public_key(fh.read())
    if not isinstance(public_key, Ed25519PublicKey):
        raise ValueError(f"Public key must be Ed25519, got {type(public_key).__name__}")

    try:
        public_key.verify(signature, payload)  # Ed25519: raises InvalidSignature if bad
    except InvalidSignature as exc:
        raise ValueError("Fingerprint signature is invalid") from exc
