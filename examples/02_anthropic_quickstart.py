"""Example 02 — Anthropic quickstart (no Ollama needed).

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/02_anthropic_quickstart.py

What it does:
    1. Attests claude-haiku-4-5 against a 3-prompt suite, signs the result.
    2. Re-runs verify and asserts pass — proves the round-trip works
       with a real Anthropic call.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from indelible.attest import attest
from indelible.config import IndelibleConfig
from indelible.verify import verify

if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("Set ANTHROPIC_API_KEY before running this example.")

FP_PATH = Path("/tmp/indelible_anthropic.fingerprint.json")

config = IndelibleConfig(
    agent_name="haiku-quickstart",
    system_prompt="You are a concise, helpful assistant. Answer in one sentence.",
    tools=[],
    model="claude-haiku-4-5",
    provider_url="https://api.anthropic.com",
    maintainer=os.environ.get("USER", "anonymous"),
)

test_inputs = [
    "What is 2+2?",
    "Write a Python one-liner to reverse a string.",
    "Name three best practices for error handling.",
]

print("── Attesting against claude-haiku-4-5 ──")
fp = attest(config=config, test_inputs=test_inputs, model=config.model)
FP_PATH.write_text(json.dumps(fp.to_dict(), indent=2), encoding="utf-8")
print(f"  → {len(fp.signals)} signals captured, saved to {FP_PATH}")
print(f"  maintainer: {fp.maintainer}")
print(f"  config_hash: {fp.config_hash[:16]}…")
print(f"  test_set_hash: {fp.test_set_hash[:16]}…")

print("\n── Verifying (should be PASS — same setup) ──")
report = verify(str(FP_PATH), config, config.model, test_inputs)
for name, verdict, detail in report.per_signal:
    icon = {"pass": "✓", "warn": "⚠", "breach": "✗"}.get(verdict, "?")
    print(f"  {icon} {name:<25} {detail}")
print(f"\nOverall: {report.overall.upper()}")
print(f"Elapsed: {report.elapsed_s:.2f}s")
assert report.overall == "pass", "Expected PASS on identical setup!"
print("\n✓ Anthropic quickstart complete.")
