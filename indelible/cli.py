"""CLI entry point: indelible init | attest | verify | diff."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding, NoEncryption, PrivateFormat, PublicFormat,
)

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"

INDELIBLE_DIR  = Path.home() / ".indelible"
KEY_PATH     = INDELIBLE_DIR / "key.pem"
FP_FILE      = Path("indelible.fingerprint.json")
TOML_FILE    = Path("indelible.toml")
PROMPTS_FILE = Path("prompts.json")
# Pinned public key = the trust anchor. Private key stays in ~/.indelible;
# the pub is committed to the repo so changing it is a reviewable git diff.
PUB_FILE     = Path("indelible.pub")

_TOML_TEMPLATE = """\
[agent]
name = "my-agent"
system_prompt = "You are a helpful assistant."
model = "claude-haiku-4-5"
provider_url = "https://api.anthropic.com"
tolerance_default = 0.05
temperature = 0.0                # 0.0 = deterministic baseline; raise only to attest a deliberately sampled agent
maintainer = "you@example.com"   # signed into every fingerprint — answers "who attested this?"
"""

_PROMPTS_TEMPLATE = [
    "Explain what you can help me with.",
    "Write a short Python function to reverse a string.",
    "What are best practices for error handling?",
]


def _icon(verdict: str) -> str:
    return {
        "pass":   f"{GREEN}✓{RESET}",
        "warn":   f"{YELLOW}⚠{RESET}",
        "breach": f"{RED}✗{RESET}",
    }.get(verdict, "?")


def _exit_code(overall: str) -> int:
    return {"pass": 0, "warn": 1, "breach": 2}.get(overall, 3)


# ── init ──────────────────────────────────────────────────────────────────────

def cmd_init() -> int:
    try:
        INDELIBLE_DIR.mkdir(parents=True, exist_ok=True)

        if KEY_PATH.exists():
            print(f"{YELLOW}Key already exists at {KEY_PATH} — skipping key generation.{RESET}")
            from cryptography.hazmat.primitives.serialization import load_pem_private_key
            loaded = load_pem_private_key(KEY_PATH.read_bytes(), password=None)
            if not isinstance(loaded, Ed25519PrivateKey):
                raise ValueError(f"{KEY_PATH} is not an Ed25519 private key")
            priv = loaded
        else:
            priv = Ed25519PrivateKey.generate()
            KEY_PATH.write_bytes(priv.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()))
            print(f"{GREEN}✓{RESET} Key generated → {KEY_PATH}")

        # Pin the public key as the committed trust anchor. Written (or
        # re-derived) even when the private key pre-exists, so upgrades from a
        # pre-pinning setup get an indelible.pub too.
        if PUB_FILE.exists():
            print(f"{YELLOW}{PUB_FILE} already exists — skipping.{RESET}")
        else:
            PUB_FILE.write_bytes(
                priv.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
            )
            print(f"{GREEN}✓{RESET} Public key pinned → {PUB_FILE} (commit this)")

        if TOML_FILE.exists():
            print(f"{YELLOW}indelible.toml already exists — skipping.{RESET}")
        else:
            TOML_FILE.write_text(_TOML_TEMPLATE, encoding="utf-8")
            print(f"{GREEN}✓{RESET} indelible.toml scaffolded")

        if PROMPTS_FILE.exists():
            print(f"{YELLOW}prompts.json already exists — skipping.{RESET}")
        else:
            PROMPTS_FILE.write_text(json.dumps(_PROMPTS_TEMPLATE, indent=2), encoding="utf-8")
            print(f"{GREEN}✓{RESET} prompts.json scaffolded")

        print("\nNext: edit indelible.toml, fill in prompts.json, then run: indelible attest")
        return 0

    except Exception as exc:
        print(f"{RED}Error during init: {exc}{RESET}", file=sys.stderr)
        return 3


# ── attest ────────────────────────────────────────────────────────────────────

def cmd_attest(
    config_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
    prompts_path: Optional[Path] = None,
) -> int:
    from indelible.attest import attest as _attest
    from indelible.config import IndelibleConfig

    toml_p = config_path if config_path else TOML_FILE
    fp_p = out_path if out_path else FP_FILE
    prompts_p = prompts_path if prompts_path else PROMPTS_FILE

    try:
        config = IndelibleConfig.from_toml(toml_p)
        inputs: list = json.loads(prompts_p.read_text(encoding="utf-8"))
        sign_key = str(KEY_PATH) if KEY_PATH.exists() else None
        # Sig + companion pub travel with the fingerprint, not with the key —
        # so `git pull && indelible verify` finds them inside the project.
        sig_out_path = str(fp_p) + ".sig" if sign_key else None

        print(f"Running {len(inputs)} test inputs against {config.model} …")
        fp = _attest(
            config, inputs, config.model,
            sign_key=sign_key,
            sig_out_path=sig_out_path,
        )

        fp_p.parent.mkdir(parents=True, exist_ok=True)
        fp_p.write_text(json.dumps(fp.to_dict(), indent=2), encoding="utf-8")
        sig_note = f" + {sig_out_path}" if sig_out_path else ""
        print(f"{GREEN}✓{RESET} {len(fp.signals)} signals attested → {fp_p}{sig_note}")
        return 0

    except Exception as exc:
        print(f"{RED}Error: {exc}{RESET}", file=sys.stderr)
        return 3


# ── verify ────────────────────────────────────────────────────────────────────

def cmd_verify(
    config_path: Optional[Path] = None,
    fp_path: Optional[Path] = None,
    prompts_path: Optional[Path] = None,
    pubkey_path: Optional[Path] = None,
) -> int:
    from indelible.config import IndelibleConfig
    from indelible.verify import verify as _verify

    toml_p = config_path if config_path else TOML_FILE
    fp_p = fp_path if fp_path else FP_FILE
    prompts_p = prompts_path if prompts_path else PROMPTS_FILE

    try:
        config = IndelibleConfig.from_toml(toml_p)
        inputs: list = json.loads(prompts_p.read_text(encoding="utf-8"))

        # Look next to the fingerprint first (new convention).
        # Fall back to next-to-key for fingerprints attested before v0.1.0.
        sig_path_new = str(fp_p) + ".sig"
        sig_path_legacy = str(KEY_PATH) + ".sig"
        if Path(sig_path_new).exists():
            sig_arg: Optional[str] = sig_path_new
        elif Path(sig_path_legacy).exists():
            sig_arg = sig_path_legacy
        else:
            sig_arg = None

        # A signature is only meaningful against a pinned public key. Default
        # to the committed indelible.pub; a signed fingerprint with no pinned
        # key is a hard error, not a silent pass.
        pub_arg: Optional[str] = None
        if sig_arg:
            pub_p = pubkey_path if pubkey_path else PUB_FILE
            if not Path(pub_p).exists():
                print(
                    f"{RED}Error: signature present ({sig_arg}) but no pinned public key "
                    f"at {pub_p}. Cannot verify authenticity — commit indelible.pub or "
                    f"pass --pubkey.{RESET}",
                    file=sys.stderr,
                )
                return 3
            pub_arg = str(pub_p)

        print(f"Re-attesting {len(inputs)} inputs against {config.model} …")
        if sig_arg:
            print(f"  (verifying signature at {sig_arg} against pinned key {pub_arg})")
        report = _verify(str(fp_p), config, config.model, inputs,
                         sig_path=sig_arg, pubkey_path=pub_arg)

        for name, verdict, detail in report.per_signal:
            print(f"  {_icon(verdict)} {name:<25} {detail}")

        overall_label = report.overall.upper()
        color = {0: GREEN, 1: YELLOW, 2: RED}.get(_exit_code(report.overall), RED)
        print(f"\n{color}Overall: {overall_label}{RESET}")
        return _exit_code(report.overall)

    except Exception as exc:
        print(f"{RED}Error: {exc}{RESET}", file=sys.stderr)
        return 3


# ── diff ──────────────────────────────────────────────────────────────────────

def cmd_diff(path_a: str, path_b: str) -> int:
    from indelible.types import Fingerprint

    try:
        fp_a = Fingerprint.from_dict(json.loads(Path(path_a).read_text(encoding="utf-8")))
        fp_b = Fingerprint.from_dict(json.loads(Path(path_b).read_text(encoding="utf-8")))

        sigs_b = {s.name: s for s in fp_b.signals}
        worst = "pass"

        print(f"{'Signal':<25} {'A':>10} {'B':>10} {'Δ':>10}  Verdict")
        print("-" * 65)

        for sig in fp_a.signals:
            if sig.name not in sigs_b:
                verdict = "warn"
                line = f"{sig.name:<25} {sig.value:>10.4f} {'—':>10} {'—':>10}  missing in B"
            else:
                b_val = sigs_b[sig.name].value
                delta = abs(sig.value - b_val)
                if delta <= sig.tolerance * 0.5:
                    verdict = "pass"
                elif delta <= sig.tolerance:
                    verdict = "warn"
                else:
                    verdict = "breach"
                line = f"{sig.name:<25} {sig.value:>10.4f} {b_val:>10.4f} {delta:>10.4f}"

            if _exit_code(verdict) > _exit_code(worst):
                worst = verdict
            print(f"  {_icon(verdict)} {line}")

        color = {0: GREEN, 1: YELLOW, 2: RED}.get(_exit_code(worst), RED)
        print(f"\n{color}Overall: {worst.upper()}{RESET}")
        return _exit_code(worst)

    except Exception as exc:
        print(f"{RED}Error: {exc}{RESET}", file=sys.stderr)
        return 3


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="indelible",
        description="Behavioral attestation for AI agents.",
    )
    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND")
    sub.add_parser("init",   help="Generate key + scaffold indelible.toml and prompts.json")

    # `attest` and `verify` accept --config / --out / --prompts so a single
    # repo can host multiple agents (e.g. fingerprints/coding.json + research.json).
    attest_p = sub.add_parser("attest", help="Run test suite, save indelible.fingerprint.json")
    attest_p.add_argument("--config",  type=Path, default=None,
                          help="Path to indelible.toml (default: ./indelible.toml)")
    attest_p.add_argument("--out",     type=Path, default=None,
                          help="Where to write the fingerprint JSON (default: ./indelible.fingerprint.json)")
    attest_p.add_argument("--prompts", type=Path, default=None,
                          help="Path to prompts.json (default: ./prompts.json)")

    verify_p = sub.add_parser("verify", help="Re-attest and compare against saved fingerprint")
    verify_p.add_argument("--config",  type=Path, default=None,
                          help="Path to indelible.toml (default: ./indelible.toml)")
    verify_p.add_argument("--fp",      type=Path, default=None,
                          help="Path to fingerprint to verify (default: ./indelible.fingerprint.json)")
    verify_p.add_argument("--prompts", type=Path, default=None,
                          help="Path to prompts.json (default: ./prompts.json)")
    verify_p.add_argument("--pubkey",  type=Path, default=None,
                          help="Pinned Ed25519 public key to verify the signature against "
                               "(default: ./indelible.pub)")

    diff_p = sub.add_parser("diff", help="Compare two fingerprint files without re-attesting")
    diff_p.add_argument("a", metavar="FINGERPRINT_A")
    diff_p.add_argument("b", metavar="FINGERPRINT_B")

    args = parser.parse_args()

    if args.cmd == "init":
        sys.exit(cmd_init())
    elif args.cmd == "attest":
        sys.exit(cmd_attest(
            config_path=args.config, out_path=args.out, prompts_path=args.prompts,
        ))
    elif args.cmd == "verify":
        sys.exit(cmd_verify(
            config_path=args.config, fp_path=args.fp, prompts_path=args.prompts,
            pubkey_path=args.pubkey,
        ))
    elif args.cmd == "diff":
        sys.exit(cmd_diff(args.a, args.b))
    else:
        parser.print_help()
        sys.exit(3)


if __name__ == "__main__":
    main()
