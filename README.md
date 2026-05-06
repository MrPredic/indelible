# bedrock-attest

> **Sigstore for AI agents.** Sign your agent's behavior today. Verify it hasn't changed tomorrow.

[![PyPI](https://img.shields.io/pypi/v/bedrock-attest)](https://pypi.org/project/bedrock-attest/)
[![Python](https://img.shields.io/pypi/pyversions/bedrock-attest)](https://pypi.org/project/bedrock-attest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/MrPredic/bedrock-attest/actions/workflows/ci.yml/badge.svg)](https://github.com/MrPredic/bedrock-attest/actions)

## Why

Model providers silently update their models. Your agent works today, but after a silent RLHF adjustment it might refuse more requests, use tools differently, or produce subtly worse outputs. You won't notice — until bedrock-attest catches it.

EU AI Act audit requirements (Q3 2026) demand "documented baselines" for High-Risk AI systems. A `bedrock.fingerprint.json` is exactly that artifact.

## Quickstart

```bash
pip install bedrock-attest
bedrock init          # generate signing key + scaffold bedrock.toml + prompts.json
bedrock attest        # run test suite, save bedrock.fingerprint.json
bedrock verify        # re-run + compare → PASS / WARN / BREACH
```

## Python API

```python
from bedrock_attest import attest, verify
from bedrock_attest.config import BedrockConfig

# attest once when you set up your agent
config = BedrockConfig.from_toml("bedrock.toml")
fp = attest(config=config, test_inputs=["prompt 1", "prompt 2"], model="claude-opus-4-7")

import json
with open("bedrock.fingerprint.json", "w") as f:
    json.dump(fp.to_dict(), f, indent=2)

# verify whenever you suspect drift or before each release
from bedrock_attest.verify import verify
report = verify("bedrock.fingerprint.json", config, "claude-opus-4-7", test_inputs=[...])
if report.breached:
    raise RuntimeError(report.summary())
```

## How it works

`bedrock attest` runs your test suite (20–50 prompts) against the model and collects these signals:

| Signal | What it measures |
|---|---|
| `refusal_rate` | Fraction of outputs matching refusal patterns |
| `latency` | P50 / P95 / mean response time |
| `vocab_entropy` | Shannon entropy of output vocabulary |
| `tool_distribution` | Histogram of tool call frequencies |
| `tool_schema_hash` | SHA-256 of canonicalized tool schemas |
| `embedding_profile` | Mean cosine similarity to centroid (`[deep]`) |
| `anchor_drift` | Cosine distance from anchor text (`[deep]`) |

The result is a **signed `bedrock.fingerprint.json`**. On `bedrock verify`, it re-runs the suite and compares signal-by-signal with configurable tolerance thresholds.

```
✓ tool_schema_hash        Δ 0.0000 (tol ±0.05)
✓ refusal_rate            Δ +0.0200 (tol ±0.10)
⚠ latency                 Δ +0.0800 (tol ±0.05)
✗ vocab_entropy           Δ +1.4200 (tol ±0.50)

Overall: BREACH
```

Exit codes: `0` = pass · `1` = warn · `2` = breach · `3` = error

## Comparison

| Category | Does | Does NOT |
|---|---|---|
| Eval frameworks (Promptfoo, DeepEval) | Quality tests per prompt | Signed behavioral artifact |
| Observability (Langfuse, Helicone) | Live tracing | Offline-verifiable contract |
| Drift detection (drift-detector, lithe) | Detects drift *after* it happens | Defines contract *before* drift |
| SLSA / Sigstore | Code provenance | Behavioral provenance |
| Snapshot tests | Exact string match | Semantic / tolerance-aware |
| **bedrock-attest** | ✓ All of the above combined | — |

## When to use

✓ After model upgrades — verify behavior hasn't regressed  
✓ CI gate — fail the build if behavioral contract is breached  
✓ EU AI Act compliance — fingerprints are "documented baselines"  
✓ Multi-provider comparison — same prompts, different models, signed diff  

## When NOT to use

✗ Real-time monitoring (use Langfuse or Helicone instead)  
✗ Testing prompt quality / correctness (use DeepEval or Promptfoo)  
✗ Detecting bugs in your own code (that's what unit tests are for)  

## Installation

```bash
# Core (no ML deps)
pip install bedrock-attest

# With semantic similarity signals
pip install "bedrock-attest[deep]"

# With drift-detector integration
pip install "bedrock-attest[drift]"

# Everything
pip install "bedrock-attest[all]"
```

## Companion projects

bedrock-attest builds on top of — and signals back to — three sibling projects:

- **[drift-detector-agent](https://github.com/MrPredic/drift-detector)** (PyPI: `drift-detector-agent`) — vocab entropy and behavioral signals. bedrock-attest uses `DriftDetectorAgent.measure_drift()` as one signal source.
- **[lithe](https://github.com/MrPredic/lithe)** — context compression with anchor-drift. bedrock-attest borrows the `anchor_drift` signal concept from `lithe.DriftMonitor`.
- **[mcp-shield](https://github.com/MrPredic/mcp-shield)** — MCP output filter. bedrock-attest uses `check_tool_definition()` logic for tool schema hashing.

## Limitations

- Format is new — not yet an RFC or standard
- Behavioral signals are statistical: identical setups may show small non-zero deltas
- Ed25519 is the default signer; Cosign (`[cosign]` extra) adds more friction
- Ollama is recommended for fast iteration (no API costs during development)

## License

MIT — Copyright 2026 MrPredic
