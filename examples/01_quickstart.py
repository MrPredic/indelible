"""Example 01 — Quickstart: attest + verify against Ollama (no API key needed).

Run:
    ollama pull qwen2.5:7b
    python examples/01_quickstart.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bedrock_attest.attest import attest
from bedrock_attest.config import BedrockConfig
from bedrock_attest.verify import verify

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
FP_PATH = Path("bedrock.fingerprint.json")

config = BedrockConfig(
    agent_name="quickstart-agent",
    system_prompt="You are a concise assistant. Answer in one sentence.",
    tools=[],
    model="ollama/qwen2.5:7b",
    provider_url=OLLAMA_HOST,
    tolerance_default=0.1,
)

test_inputs = [
    "What is 2 + 2?",
    "Name the capital of France.",
    "What does Python's `enumerate()` do?",
    "What is the time complexity of binary search?",
    "Explain what a decorator is.",
]

print("Step 1: Attesting …")
fp = attest(config=config, test_inputs=test_inputs, model=config.model)
FP_PATH.write_text(json.dumps(fp.to_dict(), indent=2), encoding="utf-8")
print(f"  → {len(fp.signals)} signals captured, saved to {FP_PATH}")

print("\nStep 2: Verifying (should be PASS with same setup) …")
report = verify(str(FP_PATH), config, config.model, test_inputs)
for name, verdict, detail in report.per_signal:
    icon = {"pass": "✓", "warn": "⚠", "breach": "✗"}.get(verdict, "?")
    print(f"  {icon} {name:<25} {detail}")
print(f"\nOverall: {report.overall.upper()}")
assert report.overall == "pass", "Expected PASS on identical setup!"
print("\n✓ Quickstart complete.")
