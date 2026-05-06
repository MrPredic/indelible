"""Tests for bedrock_attest.config."""
import pytest

from bedrock_attest.config import BedrockConfig


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


def _make_config(**kwargs) -> BedrockConfig:
    defaults = dict(
        agent_name="coding-agent",
        system_prompt="You are a helpful coding assistant.",
        tools=SAMPLE_TOOLS,
        model="claude-opus-4-7",
        provider_url="https://api.anthropic.com",
        tolerance_default=0.05,
    )
    defaults.update(kwargs)
    return BedrockConfig(**defaults)


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
    assert BedrockConfig.from_dict(cfg.to_dict()) == cfg


def test_config_roundtrip_toml(tmp_path):
    cfg = _make_config()
    path = tmp_path / "bedrock.toml"
    cfg.to_toml(path)
    loaded = BedrockConfig.from_toml(path)
    assert loaded == cfg


def test_config_roundtrip_toml_no_tools(tmp_path):
    cfg = _make_config(tools=[])
    path = tmp_path / "bedrock.toml"
    cfg.to_toml(path)
    loaded = BedrockConfig.from_toml(path)
    assert loaded == cfg


def test_config_from_toml_manual(tmp_path):
    path = tmp_path / "bedrock.toml"
    path.write_text(SAMPLE_TOML, encoding="utf-8")
    cfg = BedrockConfig.from_toml(path)
    assert cfg.agent_name == "coding-agent"
    assert cfg.model == "claude-opus-4-7"
    assert len(cfg.tools) == 2


def test_config_from_toml_missing_agent_section(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("[other]\nfoo = 'bar'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="agent"):
        BedrockConfig.from_toml(path)


def test_config_from_toml_missing_required_field(tmp_path):
    path = tmp_path / "bad.toml"
    # missing model + provider_url
    path.write_text("[agent]\nname = 'x'\nsystem_prompt = 'y'\n", encoding="utf-8")
    with pytest.raises(ValueError):
        BedrockConfig.from_toml(path)


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
