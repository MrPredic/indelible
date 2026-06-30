"""Tests for indelible.cli."""
from __future__ import annotations

import json
import subprocess
import sys
from unittest.mock import patch

import pytest

import indelible.cli as cli_mod
from indelible.config import IndelibleConfig
from indelible.types import Fingerprint, Signal


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_fp(*extra_signals: Signal) -> Fingerprint:
    base = (
        Signal("refusal_rate", 0.05, tolerance=0.1),
        Signal("latency",      0.3,  p50=0.25, p95=0.55, tolerance=0.5),
        Signal("vocab_entropy", 3.1, tolerance=1.0),
    )
    return Fingerprint(
        schema_version="1",
        config_hash="abc" * 10 + "ab",
        model="gpt-4o",
        timestamp="2026-05-06T00:00:00Z",
        maintainer="",
        signals=base + extra_signals,
        test_set_hash="def" * 10 + "de",
    )


def _make_cfg() -> IndelibleConfig:
    return IndelibleConfig(
        agent_name="test",
        system_prompt="You are helpful.",
        tools=[],
        model="gpt-4o",
        provider_url="https://api.openai.com/v1",
    )


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Redirect all CLI path constants to tmp_path."""
    monkeypatch.setattr(cli_mod, "INDELIBLE_DIR",  tmp_path)
    monkeypatch.setattr(cli_mod, "KEY_PATH",     tmp_path / "key.pem")
    monkeypatch.setattr(cli_mod, "FP_FILE",      tmp_path / "indelible.fingerprint.json")
    monkeypatch.setattr(cli_mod, "TOML_FILE",    tmp_path / "indelible.toml")
    monkeypatch.setattr(cli_mod, "PROMPTS_FILE", tmp_path / "prompts.json")
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ── cmd_init ───────────────────────────────────────────────────────────────────

def test_init_creates_key_and_files(iso):
    assert cli_mod.cmd_init() == 0
    assert (iso / "key.pem").exists()
    assert (iso / "indelible.toml").exists()
    assert (iso / "prompts.json").exists()


def test_init_toml_is_valid(iso):
    cli_mod.cmd_init()
    cfg = IndelibleConfig.from_toml(iso / "indelible.toml")
    assert cfg.agent_name  # non-empty
    assert cfg.model


def test_init_prompts_is_list(iso):
    cli_mod.cmd_init()
    prompts = json.loads((iso / "prompts.json").read_text())
    assert isinstance(prompts, list)
    assert len(prompts) >= 1


def test_init_does_not_overwrite_existing_key(iso):
    (iso / "key.pem").write_bytes(b"ORIGINAL")
    cli_mod.cmd_init()
    assert (iso / "key.pem").read_bytes() == b"ORIGINAL"


# ── cmd_diff ───────────────────────────────────────────────────────────────────

def test_diff_identical_fingerprints(iso):
    fp = _make_fp()
    (iso / "a.json").write_text(json.dumps(fp.to_dict()), encoding="utf-8")
    (iso / "b.json").write_text(json.dumps(fp.to_dict()), encoding="utf-8")
    assert cli_mod.cmd_diff(str(iso / "a.json"), str(iso / "b.json")) == 0


def test_diff_breached_fingerprints(iso):
    fp_a = _make_fp()
    extreme = tuple(
        Signal(s.name, s.value + 999.0, tolerance=s.tolerance) for s in fp_a.signals
    )
    fp_b = Fingerprint(
        schema_version=fp_a.schema_version, config_hash=fp_a.config_hash,
        model=fp_a.model, timestamp=fp_a.timestamp, maintainer=fp_a.maintainer,
        signals=extreme, test_set_hash=fp_a.test_set_hash,
    )
    (iso / "a.json").write_text(json.dumps(fp_a.to_dict()), encoding="utf-8")
    (iso / "b.json").write_text(json.dumps(fp_b.to_dict()), encoding="utf-8")
    assert cli_mod.cmd_diff(str(iso / "a.json"), str(iso / "b.json")) == 2


def test_diff_missing_file_returns_3(iso):
    (iso / "a.json").write_text(json.dumps(_make_fp().to_dict()), encoding="utf-8")
    assert cli_mod.cmd_diff(str(iso / "a.json"), str(iso / "no_such.json")) == 3


# ── cmd_attest ─────────────────────────────────────────────────────────────────

def test_attest_writes_fingerprint(iso):
    cfg = _make_cfg()
    cfg.to_toml(iso / "indelible.toml")
    (iso / "prompts.json").write_text(json.dumps(["hi", "hello"]), encoding="utf-8")

    class Stub:
        def complete(self, system, user, tools=None): return ("stub", [], 0.05)

    with patch("indelible.attest.get_provider", return_value=Stub()):
        code = cli_mod.cmd_attest()

    assert code == 0
    assert (iso / "indelible.fingerprint.json").exists()
    data = json.loads((iso / "indelible.fingerprint.json").read_text())
    assert "signals" in data


def test_attest_missing_toml_returns_3(iso):
    assert cli_mod.cmd_attest() == 3


def test_attest_with_custom_config_and_out(iso):
    """P1-5: --config / --out lets a single repo host multiple agents."""
    cfg = _make_cfg()
    custom_toml = iso / "agents" / "coding.toml"
    custom_toml.parent.mkdir(parents=True, exist_ok=True)
    cfg.to_toml(custom_toml)
    custom_prompts = iso / "agents" / "coding.prompts.json"
    custom_prompts.write_text(json.dumps(["hi"]), encoding="utf-8")
    custom_out = iso / "fingerprints" / "coding.json"

    class Stub:
        def complete(self, system, user, tools=None): return ("stub", [], 0.05)

    with patch("indelible.attest.get_provider", return_value=Stub()):
        code = cli_mod.cmd_attest(
            config_path=custom_toml, out_path=custom_out, prompts_path=custom_prompts,
        )

    assert code == 0
    assert custom_out.exists()
    # The default fingerprint must NOT have been written
    assert not (iso / "indelible.fingerprint.json").exists()


# ── cmd_verify ─────────────────────────────────────────────────────────────────

def test_verify_pass_report(iso):
    cfg = _make_cfg()
    cfg.to_toml(iso / "indelible.toml")
    inputs = ["hi", "hello"]
    (iso / "prompts.json").write_text(json.dumps(inputs), encoding="utf-8")
    fp = _make_fp()
    (iso / "indelible.fingerprint.json").write_text(json.dumps(fp.to_dict()), encoding="utf-8")

    with patch("indelible.verify.attest", return_value=fp):
        code = cli_mod.cmd_verify()

    assert code in (0, 1, 2)  # any valid exit code = no crash


def test_cli_attest_then_verify_with_signature(iso):
    """End-to-end CLI flow: init → attest (signs) → verify (checks signature)."""
    cfg = _make_cfg()
    cfg.to_toml(iso / "indelible.toml")
    (iso / "prompts.json").write_text(json.dumps(["hi", "hello"]), encoding="utf-8")

    class Stub:
        def complete(self, system, user, tools=None): return ("stub-out", [], 0.05)

    cli_mod.cmd_init()  # generates key.pem
    with patch("indelible.attest.get_provider", return_value=Stub()):
        assert cli_mod.cmd_attest() == 0
    # _sign should have written .sig + companion .pub *next to the fingerprint*,
    # not next to the key — sigs travel with the artefact (P0-2 review fix).
    assert (iso / "indelible.fingerprint.json.sig").exists()
    assert (iso / "indelible.fingerprint.json.sig.pub").exists()

    with patch("indelible.attest.get_provider", return_value=Stub()):
        code = cli_mod.cmd_verify()
    assert code == 0  # signature valid + signals deterministic → pass


# ── subprocess (main) ──────────────────────────────────────────────────────────

def test_main_help_exit0():
    result = subprocess.run(
        [sys.executable, "-m", "indelible.cli", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "indelible" in result.stdout.lower()


def test_main_no_args_exit3():
    result = subprocess.run(
        [sys.executable, "-m", "indelible.cli"],
        capture_output=True, text=True,
    )
    assert result.returncode == 3
