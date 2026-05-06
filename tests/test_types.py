"""Tests for bedrock_attest.types."""
from dataclasses import FrozenInstanceError

import pytest

from bedrock_attest.types import Fingerprint, Signal, VerifyReport


# --- Signal ---

def test_signal_minimal():
    s = Signal(name="refusal_rate", value=0.12)
    assert s.name == "refusal_rate"
    assert s.value == 0.12
    assert s.p50 is None
    assert s.tolerance == 0.05


def test_signal_all_fields():
    s = Signal("latency", 0.35, p50=0.3, p95=0.6, distribution={"fast": 0.8, "slow": 0.2}, tolerance=0.1)
    assert s.p50 == 0.3
    assert s.p95 == 0.6
    assert s.distribution == {"fast": 0.8, "slow": 0.2}
    assert s.tolerance == 0.1


def test_signal_frozen():
    s = Signal("x", 0.5)
    with pytest.raises(FrozenInstanceError):
        s.name = "y"  # type: ignore[misc]


def test_signal_roundtrip():
    s = Signal("vocab_entropy", 3.14, p50=3.0, p95=3.9, distribution={"a": 0.5, "b": 0.5})
    assert Signal.from_dict(s.to_dict()) == s


def test_signal_roundtrip_minimal():
    s = Signal("anchor_drift", 0.07)
    assert Signal.from_dict(s.to_dict()) == s


def test_signal_from_dict_defaults():
    d = {"name": "refusal_rate", "value": 0.1}
    s = Signal.from_dict(d)
    assert s.tolerance == 0.05
    assert s.p50 is None


# --- Fingerprint ---

def _make_fingerprint(**kwargs) -> Fingerprint:
    defaults = dict(
        schema_version="1",
        config_hash="abc123",
        model="claude-opus-4-7",
        timestamp="2026-05-06T00:00:00Z",
        maintainer="MrPredic",
        signals=(Signal("refusal_rate", 0.1), Signal("latency", 0.3)),
        test_set_hash="def456",
    )
    defaults.update(kwargs)
    return Fingerprint(**defaults)


def test_fingerprint_creation():
    fp = _make_fingerprint()
    assert fp.schema_version == "1"
    assert fp.model == "claude-opus-4-7"
    assert len(fp.signals) == 2


def test_fingerprint_frozen():
    fp = _make_fingerprint()
    with pytest.raises(FrozenInstanceError):
        fp.model = "gpt-4o"  # type: ignore[misc]


def test_fingerprint_roundtrip():
    fp = _make_fingerprint()
    assert Fingerprint.from_dict(fp.to_dict()) == fp


def test_fingerprint_to_dict_signals_is_list():
    fp = _make_fingerprint()
    d = fp.to_dict()
    assert isinstance(d["signals"], list)


def test_fingerprint_roundtrip_no_signals():
    fp = _make_fingerprint(signals=())
    assert Fingerprint.from_dict(fp.to_dict()) == fp


# --- VerifyReport ---

def test_verify_report_pass():
    r = VerifyReport("pass", (), 1.0)
    assert r.breached is False
    assert r.overall == "pass"


def test_verify_report_breach():
    r = VerifyReport("breach", (("refusal_rate", "breach", "Δ +33pp"),), 2.0)
    assert r.breached is True


def test_verify_report_warn():
    r = VerifyReport("warn", (("latency", "warn", "Δ +10ms"),), 1.5)
    assert not r.breached
    assert r.overall == "warn"


def test_verify_report_summary_contains_overall():
    r = VerifyReport("pass", (), 1.0)
    assert "pass" in r.summary()

    r2 = VerifyReport("breach", (("refusal_rate", "breach", "Δ +33pp"),), 2.0)
    assert "breach" in r2.summary()


def test_verify_report_summary_contains_signals():
    r = VerifyReport("warn", (("latency", "warn", "slow"),), 1.0)
    assert "latency" in r.summary()


def test_verify_report_cost_usd_optional():
    r = VerifyReport("pass", (), 0.5, cost_usd=0.002)
    assert r.cost_usd == 0.002
