"""Core data model: Signal, Fingerprint, VerifyReport."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, Type


@dataclass(frozen=True)
class Signal:
    name: str
    value: float
    p50: Optional[float] = None
    p95: Optional[float] = None
    distribution: Optional[Dict[str, float]] = None
    tolerance: float = 0.05

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "p50": self.p50,
            "p95": self.p95,
            "distribution": self.distribution,
            "tolerance": self.tolerance,
        }

    @classmethod
    def from_dict(cls: Type[Signal], d: dict) -> Signal:
        return cls(
            name=d["name"],
            value=float(d["value"]),
            p50=d.get("p50"),
            p95=d.get("p95"),
            distribution=d.get("distribution"),
            tolerance=float(d.get("tolerance", 0.05)),
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
