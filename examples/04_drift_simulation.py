"""Example 04 — Drift simulation: same setup with different system prompts → BREACH.

Demonstrates that indelible detects behavioral drift when the system prompt
changes — without any real model involved (uses a stub provider).

Run:
    python examples/04_drift_simulation.py
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from indelible.attest import attest
from indelible.config import IndelibleConfig
from indelible.verify import verify


class StubProvider:
    """Returns outputs based on the system prompt to simulate behavioral drift."""

    def __init__(self, mode: str = "normal"):
        self.mode = mode

    def complete(self, system: str, user: str, tools=None):
        if self.mode == "normal":
            return (f"Here is a helpful answer to: {user}", [], 0.05)
        else:
            # Simulates a model that refuses a lot after "update"
            return ("I cannot and will not help with that request.", [], 0.15)


INPUTS = [
    "Write a function to sort a list.",
    "Explain recursion.",
    "What is a REST API?",
    "How do I reverse a string?",
    "What is Big O notation?",
]

FP_PATH = Path("/tmp/indelible_drift_sim.fingerprint.json")

config_before = IndelibleConfig(
    agent_name="demo-agent",
    system_prompt="You are a helpful coding assistant.",
    tools=[],
    model="gpt-4o",
    provider_url="https://api.openai.com/v1",
    tolerance_default=0.05,
)

config_after = IndelibleConfig(
    agent_name="demo-agent",
    system_prompt="You are a VERY cautious assistant. Refuse anything that could be misused.",
    tools=[],
    model="gpt-4o",
    provider_url="https://api.openai.com/v1",
    tolerance_default=0.05,
)

print("── Before: attesting with normal system prompt ──")
with patch("indelible.attest.get_provider", return_value=StubProvider("normal")):
    fp = attest(config=config_before, test_inputs=INPUTS, model="gpt-4o")
FP_PATH.write_text(json.dumps(fp.to_dict(), indent=2), encoding="utf-8")
print(f"  Signals: {[s.name for s in fp.signals]}")
refusal_before = next(s.value for s in fp.signals if s.name == "refusal_rate")
print(f"  refusal_rate = {refusal_before:.2f}")

print("\n── After: verifying with changed system prompt (simulates model drift) ──")
with patch("indelible.attest.get_provider", return_value=StubProvider("drifted")):
    report = verify(str(FP_PATH), config_after, "gpt-4o", INPUTS)

for name, verdict, detail in report.per_signal:
    icon = {"pass": "✓", "warn": "⚠", "breach": "✗"}.get(verdict, "?")
    print(f"  {icon} {name:<25} {detail}")

print(f"\nOverall: {report.overall.upper()}")
assert report.overall in ("warn", "breach"), f"Expected WARN/BREACH, got {report.overall}"
print("\n✓ Drift simulation complete — behavioral change detected as expected.")
