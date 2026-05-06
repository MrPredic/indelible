"""Tests for bedrock_attest.attest and bedrock_attest.verify."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding, NoEncryption, PrivateFormat, PublicFormat,
)

from bedrock_attest.attest import attest
from bedrock_attest.config import BedrockConfig
from bedrock_attest.types import Fingerprint, Signal
from bedrock_attest.verify import verify


class StubProvider:
    """Deterministic stub — always returns same output."""
    def complete(self, system: str, user: str, tools=None):
        return ("The answer is: " + user, [], 0.05)


@pytest.fixture
def cfg():
    return BedrockConfig(
        agent_name="test-agent",
        system_prompt="You are a helpful assistant.",
        tools=[{"name": "search", "description": "search the web"}],
        model="gpt-4o",
        provider_url="https://api.openai.com/v1",
    )


@pytest.fixture
def inputs():
    return ["What is 2+2?", "What is the capital of France?", "Explain recursion."]


def _stub_attest(cfg, inputs):
    with patch("bedrock_attest.attest.get_provider", return_value=StubProvider()):
        return attest(cfg, inputs, "gpt-4o")


def _save_fp(fp: Fingerprint, path: Path) -> Path:
    fpath = path / "bedrock.fingerprint.json"
    fpath.write_text(json.dumps(fp.to_dict()), encoding="utf-8")
    return fpath


# --- attest() ---

def test_attest_returns_fingerprint(cfg, inputs):
    fp = _stub_attest(cfg, inputs)
    assert isinstance(fp, Fingerprint)


def test_attest_model_stored(cfg, inputs):
    fp = _stub_attest(cfg, inputs)
    assert fp.model == "gpt-4o"


def test_attest_config_hash_matches(cfg, inputs):
    fp = _stub_attest(cfg, inputs)
    assert fp.config_hash == cfg.canonical_hash()


def test_attest_schema_version(cfg, inputs):
    fp = _stub_attest(cfg, inputs)
    assert fp.schema_version == "1"


def test_attest_has_signals(cfg, inputs):
    fp = _stub_attest(cfg, inputs)
    assert len(fp.signals) > 0
    assert all(isinstance(s, Signal) for s in fp.signals)


def test_attest_signal_names_include_core(cfg, inputs):
    fp = _stub_attest(cfg, inputs)
    names = {s.name for s in fp.signals}
    assert "refusal_rate" in names
    assert "latency" in names
    assert "vocab_entropy" in names


def test_attest_empty_inputs_raises(cfg):
    with patch("bedrock_attest.attest.get_provider", return_value=StubProvider()):
        with pytest.raises(ValueError, match="non-empty"):
            attest(cfg, [], "gpt-4o")


def test_attest_wrong_config_type_raises():
    with patch("bedrock_attest.attest.get_provider", return_value=StubProvider()):
        with pytest.raises(ValueError):
            attest("not-a-config", ["hi"], "gpt-4o")  # type: ignore


def test_attest_does_not_mutate_inputs(cfg):
    original = ["prompt A", "prompt B"]
    copy_ = list(original)
    with patch("bedrock_attest.attest.get_provider", return_value=StubProvider()):
        attest(cfg, original, "gpt-4o")
    assert original == copy_


def test_attest_sign_writes_sig_file(cfg, inputs, tmp_path):
    key = Ed25519PrivateKey.generate()
    key_path = tmp_path / "key.pem"
    key_path.write_bytes(key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()))

    with patch("bedrock_attest.attest.get_provider", return_value=StubProvider()):
        attest(cfg, inputs, "gpt-4o", sign_key=str(key_path))

    assert (tmp_path / "key.pem.sig").exists()


# --- verify() ---

def test_verify_pass_deterministic(cfg, inputs, tmp_path):
    """Same config + same stub → deterministic signals → overall pass."""
    fp = _stub_attest(cfg, inputs)
    fpath = _save_fp(fp, tmp_path)

    with patch("bedrock_attest.attest.get_provider", return_value=StubProvider()):
        report = verify(str(fpath), cfg, "gpt-4o", inputs)

    assert report.overall == "pass"
    assert report.elapsed_s >= 0.0


def test_verify_breach_on_different_values(cfg, inputs, tmp_path):
    """Fingerprint with extreme values against normal stub → breach."""
    fp = _stub_attest(cfg, inputs)
    # Build a fingerprint with signals having very different values
    extreme_signals = tuple(
        Signal(name=s.name, value=s.value + 999.0, tolerance=s.tolerance)
        for s in fp.signals
    )
    extreme_fp = Fingerprint(
        schema_version=fp.schema_version,
        config_hash=fp.config_hash,
        model=fp.model,
        timestamp=fp.timestamp,
        maintainer=fp.maintainer,
        signals=extreme_signals,
        test_set_hash=fp.test_set_hash,
    )
    fpath = _save_fp(extreme_fp, tmp_path)

    with patch("bedrock_attest.attest.get_provider", return_value=StubProvider()):
        report = verify(str(fpath), cfg, "gpt-4o", inputs)

    assert report.overall == "breach"


def test_verify_per_signal_populated(cfg, inputs, tmp_path):
    fp = _stub_attest(cfg, inputs)
    fpath = _save_fp(fp, tmp_path)

    with patch("bedrock_attest.attest.get_provider", return_value=StubProvider()):
        report = verify(str(fpath), cfg, "gpt-4o", inputs)

    assert len(report.per_signal) > 0
    for name, verdict, detail in report.per_signal:
        assert verdict in ("pass", "warn", "breach")


def test_verify_missing_signal_is_warn(cfg, inputs, tmp_path):
    """A signal in the saved fingerprint that is absent from fresh run → warn."""
    fp = _stub_attest(cfg, inputs)
    # Inject a ghost signal that no collector will produce
    ghost = Signal(name="ghost_signal_xyz", value=0.5, tolerance=0.01)
    augmented = Fingerprint(
        schema_version=fp.schema_version,
        config_hash=fp.config_hash,
        model=fp.model,
        timestamp=fp.timestamp,
        maintainer=fp.maintainer,
        signals=fp.signals + (ghost,),
        test_set_hash=fp.test_set_hash,
    )
    fpath = _save_fp(augmented, tmp_path)

    with patch("bedrock_attest.attest.get_provider", return_value=StubProvider()):
        report = verify(str(fpath), cfg, "gpt-4o", inputs)

    ghost_verdict = next(v for n, v, _ in report.per_signal if n == "ghost_signal_xyz")
    assert ghost_verdict == "warn"


def test_verify_invalid_sig_raises(cfg, inputs, tmp_path):
    fp = _stub_attest(cfg, inputs)
    fpath = _save_fp(fp, tmp_path)
    sig_path = tmp_path / "fake.sig"
    sig_path.write_bytes(b"not-a-real-signature")
    pub_path = tmp_path / "fake.pub"
    key = Ed25519PrivateKey.generate()
    pub_path.write_bytes(key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo))

    with patch("bedrock_attest.attest.get_provider", return_value=StubProvider()):
        with pytest.raises(ValueError, match="[Ss]ignature"):
            verify(str(fpath), cfg, "gpt-4o", inputs, sig_path=str(sig_path))
