"""Tests for indelible.attest and indelible.verify."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding, NoEncryption, PrivateFormat, PublicFormat,
)

from indelible.attest import attest
from indelible.config import IndelibleConfig
from indelible.types import Fingerprint, Signal
from indelible.verify import verify


class StubProvider:
    """Deterministic stub — always returns same output."""
    def complete(self, system: str, user: str, tools=None):
        return ("The answer is: " + user, [], 0.05)


@pytest.fixture
def cfg():
    return IndelibleConfig(
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
    with patch("indelible.attest.get_provider", return_value=StubProvider()):
        return attest(cfg, inputs, "gpt-4o")


def _save_fp(fp: Fingerprint, path: Path) -> Path:
    fpath = path / "indelible.fingerprint.json"
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
    with patch("indelible.attest.get_provider", return_value=StubProvider()):
        with pytest.raises(ValueError, match="non-empty"):
            attest(cfg, [], "gpt-4o")


def test_attest_wrong_config_type_raises():
    with patch("indelible.attest.get_provider", return_value=StubProvider()):
        with pytest.raises(ValueError):
            attest("not-a-config", ["hi"], "gpt-4o")  # type: ignore


def test_api_key_groq_env(monkeypatch):
    """GROQ_API_KEY must be picked up for non-Anthropic non-Ollama models."""
    from indelible.attest import _api_key
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    assert _api_key("llama-3.3-70b-versatile") == "gsk-test"


def test_api_key_anthropic_env(monkeypatch):
    from indelible.attest import _api_key
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert _api_key("claude-opus-4-7") == "sk-ant-test"


def test_api_key_ollama_is_none(monkeypatch):
    from indelible.attest import _api_key
    assert _api_key("ollama/qwen2.5:7b") is None


def test_api_key_picks_groq_when_url_is_groq(monkeypatch):
    """Both OPENAI + GROQ keys set; URL host says Groq → return GROQ key.
    Pre-fix bug: iteration order made OPENAI win regardless of provider_url."""
    from indelible.attest import _api_key
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    assert _api_key("llama-3.3-70b-versatile",
                    provider_url="https://api.groq.com/openai/v1") == "groq-key"


def test_api_key_picks_together_when_url_is_together(monkeypatch):
    from indelible.attest import _api_key
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("TOGETHER_API_KEY", "together-key")
    assert _api_key("llama-3-70b",
                    provider_url="https://api.together.xyz/v1") == "together-key"


def test_api_key_falls_back_when_specific_unset(monkeypatch):
    """URL says Groq but only OPENAI key is set → fall back to OPENAI."""
    from indelible.attest import _api_key
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    assert _api_key("llama-3", provider_url="https://api.groq.com/v1") == "openai-key"


def test_attest_does_not_mutate_inputs(cfg):
    original = ["prompt A", "prompt B"]
    copy_ = list(original)
    with patch("indelible.attest.get_provider", return_value=StubProvider()):
        attest(cfg, original, "gpt-4o")
    assert original == copy_


def test_attest_sign_writes_sig_and_pub(cfg, inputs, tmp_path):
    key = Ed25519PrivateKey.generate()
    key_path = tmp_path / "key.pem"
    key_path.write_bytes(key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()))

    with patch("indelible.attest.get_provider", return_value=StubProvider()):
        attest(cfg, inputs, "gpt-4o", sign_key=str(key_path))

    assert (tmp_path / "key.pem.sig").exists()
    assert (tmp_path / "key.pem.sig.pub").exists()  # companion pub sits next to .sig


# --- verify() ---

def test_verify_pass_deterministic(cfg, inputs, tmp_path):
    """Same config + same stub → deterministic signals → overall pass."""
    fp = _stub_attest(cfg, inputs)
    fpath = _save_fp(fp, tmp_path)

    with patch("indelible.attest.get_provider", return_value=StubProvider()):
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

    with patch("indelible.attest.get_provider", return_value=StubProvider()):
        report = verify(str(fpath), cfg, "gpt-4o", inputs)

    assert report.overall == "breach"


def test_verify_per_signal_populated(cfg, inputs, tmp_path):
    fp = _stub_attest(cfg, inputs)
    fpath = _save_fp(fp, tmp_path)

    with patch("indelible.attest.get_provider", return_value=StubProvider()):
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

    with patch("indelible.attest.get_provider", return_value=StubProvider()):
        report = verify(str(fpath), cfg, "gpt-4o", inputs)

    ghost_verdict = next(v for n, v, _ in report.per_signal if n == "ghost_signal_xyz")
    assert ghost_verdict == "warn"


def test_sign_then_verify_roundtrip(cfg, inputs, tmp_path):
    """End-to-end: attest with sign_key, then verify with the produced .sig — must pass."""
    key = Ed25519PrivateKey.generate()
    key_path = tmp_path / "key.pem"
    key_path.write_bytes(key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()))

    with patch("indelible.attest.get_provider", return_value=StubProvider()):
        fp = attest(cfg, inputs, "gpt-4o", sign_key=str(key_path))
    fpath = _save_fp(fp, tmp_path)

    sig_path = str(key_path) + ".sig"
    with patch("indelible.attest.get_provider", return_value=StubProvider()):
        report = verify(str(fpath), cfg, "gpt-4o", inputs, sig_path=sig_path)

    assert report.overall == "pass"


def test_verify_breach_on_wrong_config(cfg, inputs, tmp_path):
    """Verify against a DIFFERENT config than the one used to attest → must breach."""
    fp = _stub_attest(cfg, inputs)
    fpath = _save_fp(fp, tmp_path)

    other_cfg = IndelibleConfig(
        agent_name=cfg.agent_name,
        system_prompt="A COMPLETELY DIFFERENT SYSTEM PROMPT",  # config_hash now differs
        tools=cfg.tools, model=cfg.model, provider_url=cfg.provider_url,
    )
    with patch("indelible.attest.get_provider", return_value=StubProvider()):
        report = verify(str(fpath), other_cfg, "gpt-4o", inputs)

    assert report.overall == "breach"
    assert any(name == "config_hash" and v == "breach" for name, v, _ in report.per_signal)


def test_verify_breach_on_wrong_test_inputs(cfg, inputs, tmp_path):
    """Verify with DIFFERENT test inputs than original → must breach (test_set_hash mismatch)."""
    fp = _stub_attest(cfg, inputs)
    fpath = _save_fp(fp, tmp_path)

    other_inputs = ["totally different prompts", "that werent in the original set"]
    with patch("indelible.attest.get_provider", return_value=StubProvider()):
        report = verify(str(fpath), cfg, "gpt-4o", other_inputs)

    assert report.overall == "breach"
    assert any(name == "test_set_hash" and v == "breach" for name, v, _ in report.per_signal)


def test_verify_unexpected_fresh_signal_warns(cfg, inputs, tmp_path):
    """A signal in the fresh run but missing from the saved fingerprint
    must surface as `warn` — schema-drift visibility."""
    fp = _stub_attest(cfg, inputs)
    # Drop one signal from the saved fingerprint to simulate "fresh has more"
    pruned = tuple(s for s in fp.signals if s.name != "vocab_entropy")
    pruned_fp = Fingerprint(
        schema_version=fp.schema_version, config_hash=fp.config_hash,
        model=fp.model, timestamp=fp.timestamp, maintainer=fp.maintainer,
        signals=pruned, test_set_hash=fp.test_set_hash,
    )
    fpath = _save_fp(pruned_fp, tmp_path)

    with patch("indelible.attest.get_provider", return_value=StubProvider()):
        report = verify(str(fpath), cfg, "gpt-4o", inputs)

    # Find the warn entry for vocab_entropy in fresh-only-set
    matches = [(n, v, d) for n, v, d in report.per_signal if n == "vocab_entropy"]
    assert any(v == "warn" for _, v, _ in matches)


def test_verify_digest_signal_breaches_on_mismatch(cfg, inputs, tmp_path):
    """Regression: tool_schema_hash uses Signal.digest (exact match).
    A different digest must breach, not pass via tolerance band — even when
    the numeric values are identical (both 0.0).
    """
    fp = _stub_attest(cfg, inputs)
    # Tamper the tool_schema_hash digest in the saved fingerprint
    tampered = tuple(
        Signal(name=s.name, value=s.value, tolerance=s.tolerance,
               distribution=s.distribution, digest="0" * 64)
        if s.name == "tool_schema_hash" else s
        for s in fp.signals
    )
    tampered_fp = Fingerprint(
        schema_version=fp.schema_version, config_hash=fp.config_hash,
        model=fp.model, timestamp=fp.timestamp, maintainer=fp.maintainer,
        signals=tampered, test_set_hash=fp.test_set_hash,
    )
    fpath = _save_fp(tampered_fp, tmp_path)

    with patch("indelible.attest.get_provider", return_value=StubProvider()):
        report = verify(str(fpath), cfg, "gpt-4o", inputs)

    assert report.overall == "breach"
    schema_verdict = next(v for n, v, _ in report.per_signal if n == "tool_schema_hash")
    assert schema_verdict == "breach"


def test_attest_writes_maintainer_from_config(inputs, tmp_path):
    """[agent].maintainer must be propagated into the signed Fingerprint."""
    cfg = IndelibleConfig(
        agent_name="t", system_prompt="s",
        tools=[], model="gpt-4o", provider_url="https://api.openai.com/v1",
        maintainer="alice@example.com",
    )
    with patch("indelible.attest.get_provider", return_value=StubProvider()):
        fp = attest(cfg, inputs, "gpt-4o")
    assert fp.maintainer == "alice@example.com"


def test_verify_short_circuits_on_config_hash_mismatch(cfg, inputs, tmp_path):
    """When config_hash mismatches, verify must NOT call the live provider —
    saves API budget on a verdict that's already breach."""
    fp = _stub_attest(cfg, inputs)
    fpath = _save_fp(fp, tmp_path)

    other_cfg = IndelibleConfig(
        agent_name=cfg.agent_name,
        system_prompt="DIFFERENT",
        tools=cfg.tools, model=cfg.model, provider_url=cfg.provider_url,
    )
    # If short-circuit works, get_provider must NOT be called
    from unittest.mock import MagicMock
    fake_provider = MagicMock()
    with patch("indelible.attest.get_provider", return_value=fake_provider):
        report = verify(str(fpath), other_cfg, "gpt-4o", inputs)
    assert report.overall == "breach"
    fake_provider.complete.assert_not_called()


def test_verify_invalid_sig_raises(cfg, inputs, tmp_path):
    fp = _stub_attest(cfg, inputs)
    fpath = _save_fp(fp, tmp_path)
    sig_path = tmp_path / "fake.sig"
    sig_path.write_bytes(b"not-a-real-signature")
    # companion pub key lives at sig_path + ".pub" (new convention)
    pub_path = Path(str(sig_path) + ".pub")
    key = Ed25519PrivateKey.generate()
    pub_path.write_bytes(key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo))

    with patch("indelible.attest.get_provider", return_value=StubProvider()):
        with pytest.raises(ValueError, match="[Ss]ignature"):
            verify(str(fpath), cfg, "gpt-4o", inputs, sig_path=str(sig_path))
