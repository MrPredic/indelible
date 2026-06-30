"""Core data model: Signal, Fingerprint, VerifyReport."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, Type


@dataclass(frozen=True)
class Signal:
    name: str
    value: float
    p50: Optional[float] = None
    p95: Optional[float] = None
    distribution: Optional[Dict[str, float]] = None
    # 0.05 is a defensive floor for hand-built Signals; built-in collectors
    # set their own per-signal tolerance (see docs/TOLERANCES.md):
    # latency=0.30, refusal_rate=0.10, vocab_entropy=0.50, tool_distribution=0.10,
    # tool_schema_hash=0.0 (uses digest), embedding_profile=0.05, anchor_drift=0.05.
    tolerance: float = 0.05
    # Optional exact-equality digest. When set, verify uses string-equal
    # comparison instead of numeric tolerance — required for SHA-style
    # signals like tool_schema_hash where any bit-flip must be a breach.
    digest: Optional[str] = None

    def __post_init__(self) -> None:
        # Coerce numerics to float so JSON roundtrip is byte-stable
        # (Signal(value=5) and Signal(value=5.0) must produce identical signing bytes).
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "tolerance", float(self.tolerance))
        if self.p50 is not None:
            object.__setattr__(self, "p50", float(self.p50))
        if self.p95 is not None:
            object.__setattr__(self, "p95", float(self.p95))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "p50": self.p50,
            "p95": self.p95,
            "distribution": self.distribution,
            "tolerance": self.tolerance,
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls: Type[Signal], d: dict) -> Signal:
        tol = d.get("tolerance")
        return cls(
            name=d["name"],
            value=float(d["value"]),
            p50=d.get("p50"),
            p95=d.get("p95"),
            distribution=d.get("distribution"),
            tolerance=float(tol) if tol is not None else 0.05,
            digest=d.get("digest"),
        )


@dataclass(frozen=True)
class Fingerprint:
    schema_version: str
    config_hash: str
    model: str
    timestamp: str
    maintainer: str
    signals: Tuple[Signal, ...]
    test_set_hash: str

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "config_hash": self.config_hash,
            "model": self.model,
            "timestamp": self.timestamp,
            "maintainer": self.maintainer,
            "signals": [s.to_dict() for s in self.signals],
            "test_set_hash": self.test_set_hash,
        }

    def canonical_bytes(self) -> bytes:
        """Single source of truth for the signing/verification payload.

        Both ``attest._sign()`` and ``verify._verify_signature()`` must use
        this — otherwise a future schema change could silently desync them.
        """
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False).encode()

    def canonical_digest(self) -> str:
        """SHA-256 hex digest of ``canonical_bytes()`` — useful for testing/diffing."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls: Type[Fingerprint], d: dict) -> Fingerprint:
        return cls(
            schema_version=d["schema_version"],
            config_hash=d["config_hash"],
            model=d["model"],
            timestamp=d["timestamp"],
            maintainer=d["maintainer"],
            signals=tuple(Signal.from_dict(s) for s in d["signals"]),
            test_set_hash=d["test_set_hash"],
        )


@dataclass(frozen=True)
class VerifyReport:
    overall: Literal["pass", "warn", "breach"]
    per_signal: Tuple[Tuple[str, str, str], ...]
    elapsed_s: float
    cost_usd: Optional[float] = None

    @property
    def breached(self) -> bool:
        return self.overall == "breach"

    def summary(self) -> str:
        details = ", ".join(f"{name}: {verdict}" for name, verdict, _ in self.per_signal)
        return f"Verification {self.overall}" + (f" — {details}" if details else "")
