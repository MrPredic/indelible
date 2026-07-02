"""Tests for indelible.config."""
import pytest

from indelible.config import IndelibleConfig


SAMPLE_TOOLS = [
    {"name": "read_file", "description": "Reads a file"},
    {"name": "write_file", "description": "Writes a file"},
]

SAMPLE_TOML = """\
[agent]
name = "coding-agent"
system_prompt = "You are a helpful coding assistant."
model = "claude-opus-4-7"
provider_url = "https://api.anthropic.com"
tolerance_default = 0.05

[[tools]]
name = "read_file"
description = "Reads a file"

[[tools]]
name = "write_file"
description = "Writes a file"
"""


def _make_config(**kwargs) -> IndelibleConfig:
    defaults = dict(
        agent_name="coding-agent",
        system_prompt="You are a helpful coding assistant.",
        tools=SAMPLE_TOOLS,
        model="claude-opus-4-7",
        provider_url="https://api.anthropic.com",
        tolerance_default=0.05,
    )
    defaults.update(kwargs)
    return IndelibleConfig(**defaults)


def test_config_creation():
    cfg = _make_config()
    assert cfg.agent_name == "coding-agent"
    assert cfg.model == "claude-opus-4-7"
    assert len(cfg.tools) == 2


def test_config_tools_deepcopied():
    tools = [{"name": "t1"}]
    cfg = _make_config(tools=tools)
    tools[0]["name"] = "mutated"
    assert cfg.tools[0]["name"] == "t1"


def test_config_equality():
    assert _make_config() == _make_config()


def test_config_inequality():
    assert _make_config(model="gpt-4o") != _make_config(model="claude-opus-4-7")


def test_config_roundtrip_dict():
    cfg = _make_config()
    assert IndelibleConfig.from_dict(cfg.to_dict()) == cfg


def test_config_roundtrip_toml(tmp_path):
    cfg = _make_config()
    path = tmp_path / "indelible.toml"
    cfg.to_toml(path)
    loaded = IndelibleConfig.from_toml(path)
    assert loaded == cfg


def test_config_roundtrip_toml_no_tools(tmp_path):
    cfg = _make_config(tools=[])
    path = tmp_path / "indelible.toml"
    cfg.to_toml(path)
    loaded = IndelibleConfig.from_toml(path)
    assert loaded == cfg


def test_config_from_toml_manual(tmp_path):
    path = tmp_path / "indelible.toml"
    path.write_text(SAMPLE_TOML, encoding="utf-8")
    cfg = IndelibleConfig.from_toml(path)
    assert cfg.agent_name == "coding-agent"
    assert cfg.model == "claude-opus-4-7"
    assert len(cfg.tools) == 2


def test_config_from_toml_missing_agent_section(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("[other]\nfoo = 'bar'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="agent"):
        IndelibleConfig.from_toml(path)


def test_config_from_toml_missing_required_field(tmp_path):
    path = tmp_path / "bad.toml"
    # missing model + provider_url
    path.write_text("[agent]\nname = 'x'\nsystem_prompt = 'y'\n", encoding="utf-8")
    with pytest.raises(ValueError):
        IndelibleConfig.from_toml(path)


def test_canonical_hash_deterministic():
    cfg1 = _make_config()
    cfg2 = _make_config()
    assert cfg1.canonical_hash() == cfg2.canonical_hash()


def test_canonical_hash_changes_with_prompt():
    cfg1 = _make_config(system_prompt="Prompt A")
    cfg2 = _make_config(system_prompt="Prompt B")
    assert cfg1.canonical_hash() != cfg2.canonical_hash()


def test_canonical_hash_is_sha256():
    cfg = _make_config()
    h = cfg.canonical_hash()
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# --- maintainer wiring (P0-4) ---

def test_maintainer_default_empty():
    cfg = _make_config()
    assert cfg.maintainer == ""


def test_maintainer_excluded_from_canonical_hash():
    """Maintainer identifies WHO attested, not WHAT — must not invalidate
    fingerprints when the on-call rotation changes."""
    a = _make_config(maintainer="alice@x.com")
    b = _make_config(maintainer="bob@x.com")
    assert a.canonical_hash() == b.canonical_hash()


def test_maintainer_roundtrip_toml(tmp_path):
    cfg = _make_config(maintainer="alice@example.com")
    path = tmp_path / "indelible.toml"
    cfg.to_toml(path)
    assert IndelibleConfig.from_toml(path).maintainer == "alice@example.com"


# --- temperature wiring (determinism fix) ---

def test_temperature_default_zero():
    """Default temperature must be 0.0 — deterministic baseline minimises
    run-to-run sampling noise that would otherwise cause false breaches."""
    assert _make_config().temperature == 0.0


def test_temperature_roundtrip_toml(tmp_path):
    cfg = _make_config(temperature=0.7)
    path = tmp_path / "indelible.toml"
    cfg.to_toml(path)
    assert IndelibleConfig.from_toml(path).temperature == 0.7


def test_temperature_in_canonical_hash():
    """Different sampling temperature = different behavioral baseline →
    canonical_hash MUST differ (temp 0 vs temp 1 produce different
    vocab_entropy/refusal distributions)."""
    a = _make_config(temperature=0.0)
    b = _make_config(temperature=1.0)
    assert a.canonical_hash() != b.canonical_hash()


def test_temperature_default_keeps_legacy_hash_stable():
    """temperature=0.0 (the default) must hash identically whether set
    explicitly or omitted — but note this DOES change the pre-temperature
    canonical form, so existing fingerprints re-attest under schema."""
    a = _make_config(temperature=0.0)
    b = _make_config()
    assert a.canonical_hash() == b.canonical_hash()


# --- refusal_patterns wiring (P1-9) ---

def test_refusal_patterns_default_none():
    assert _make_config().refusal_patterns is None


def test_refusal_patterns_roundtrip_toml(tmp_path):
    custom = ["je ne peux pas", "no puedo"]
    cfg = _make_config(refusal_patterns=custom)
    path = tmp_path / "indelible.toml"
    cfg.to_toml(path)
    assert IndelibleConfig.from_toml(path).refusal_patterns == custom


def test_refusal_patterns_in_canonical_hash():
    """Different refusal patterns = different attested baseline →
    canonical_hash MUST differ (otherwise verify could pass with patterns
    that wouldn't match the original attest)."""
    a = _make_config(refusal_patterns=["I cannot"])
    b = _make_config(refusal_patterns=["je ne peux pas"])
    assert a.canonical_hash() != b.canonical_hash()


def test_refusal_patterns_none_keeps_legacy_hash_stable():
    """Configs that don't set refusal_patterns must hash identically to
    pre-P1 configs — backward compat for existing fingerprints."""
    a = _make_config()  # refusal_patterns=None
    a.refusal_patterns = None
    b = _make_config()
    assert a.canonical_hash() == b.canonical_hash()
