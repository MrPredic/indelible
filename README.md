# indelible

> **Detect when your AI agent silently drifts.** indelible captures a signed behavioral fingerprint (refusal rate, latency, vocab entropy, tool-use distribution) and re-runs it on demand — your CI gate for model-update day.

[![PyPI](https://img.shields.io/pypi/v/indelible)](https://pypi.org/project/indelible/)
[![Python](https://img.shields.io/pypi/pyversions/indelible)](https://pypi.org/project/indelible/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/MrPredic/indelible/actions/workflows/ci.yml/badge.svg)](https://github.com/MrPredic/indelible/actions)

## Why

Model providers silently update their models. Your agent works today, but after a silent RLHF adjustment it might refuse more requests, use tools differently, or produce subtly worse outputs. You won't notice — until indelible catches it.

The output is a single signed JSON file (`indelible.fingerprint.json`) — diffable, offline-verifiable against a pinned key, and re-runnable across providers (Anthropic, OpenAI, Ollama, Groq). It can also serve as one supporting artefact in an EU AI Act High-Risk record-keeping file — a dated, signed behavioral baseline (not a certified deliverable on its own).

## Quickstart

```bash
pip install indelible
indelible init          # generate signing key + pinned indelible.pub + scaffold indelible.toml + prompts.json
indelible attest        # run test suite (temperature=0), save + sign indelible.fingerprint.json
indelible verify        # re-run + compare, check signature vs pinned indelible.pub → PASS / WARN / BREACH
```

> **Commit `indelible.pub`.** It's the trust anchor `verify` checks the signature against. The private key stays in `~/.indelible/key.pem` and never enters the repo.

## Python API

```python
from indelible import attest, verify
from indelible.config import IndelibleConfig

# attest once when you set up your agent
config = IndelibleConfig.from_toml("indelible.toml")
fp = attest(config=config, test_inputs=["prompt 1", "prompt 2"], model="claude-opus-4-7")

import json
with open("indelible.fingerprint.json", "w") as f:
    json.dump(fp.to_dict(), f, indent=2)

# verify whenever you suspect drift or before each release
from indelible.verify import verify
report = verify("indelible.fingerprint.json", config, "claude-opus-4-7", test_inputs=[...])
if report.breached:
    raise RuntimeError(report.summary())
```

## How it works

`indelible attest` runs your test suite (20–50 prompts) against the model and collects these signals:

| Signal | What it measures |
|---|---|
| `refusal_rate` | Fraction of outputs matching refusal patterns |
| `latency` | P50 / P95 / mean response time |
| `vocab_entropy` | Shannon entropy of output vocabulary |
| `tool_distribution` | Histogram of tool call frequencies |
| `tool_schema_hash` | SHA-256 of canonicalized tool schemas |
| `embedding_profile` | Mean cosine similarity to centroid (`[deep]`) |
| `anchor_drift` | Cosine distance from anchor text (`[deep]`) |

The result is a **signed `indelible.fingerprint.json`**. On `indelible verify`, it re-runs the suite and compares signal-by-signal with configurable tolerance thresholds.

```
✓ tool_schema_hash        Δ 0.0000 (tol ±0.05)
✓ refusal_rate            Δ +0.0200 (tol ±0.10)
⚠ latency                 Δ +0.0800 (tol ±0.05)
✗ vocab_entropy           Δ +1.4200 (tol ±0.50)

Overall: BREACH
```

Exit codes: `0` = pass · `1` = warn · `2` = breach · `3` = error

## Where it sits in the landscape

```
                    behavioral signals
                          ▲
                          │
   Galileo Signals  ●     │     ● indelible
   Arize Phoenix    ●     │
   WhyLabs          ●     │     ● Promptfoo (eval)
   ─────────runtime───────┼──────CI──────────►
                          │     ● DeepEval
   Helicone (logs)  ●     │     ● Sigstore
                          │     ● sigstore-a2a
                          │     ● in-toto / SLSA
                          │
                     static signals (code/identity)
```

indelible occupies the `(behavioral, CI)` quadrant. Adjacent tools each cover *part* of the surface, but none ship a signed, offline-verifiable behavioral artefact you can diff across model upgrades.

| Tool | Surface | What indelible adds |
|---|---|---|
| **Promptfoo, DeepEval** | CI quality tests on prompt/response | Behavioral consistency over time, signed artefact |
| **Arize Phoenix, WhyLabs** | Runtime observability + drift dashboards | Offline JSON you can sign, diff, and gate CI on |
| **Sigstore, in-toto, SLSA** | Sign code/identity/build provenance | Sign *behavioral* properties of an LLM agent |
| **drift-detector** | Detect drift after it happens | Define the contract *before* drift, fail the build |

These are complements, not competitors. Use Promptfoo for quality, Phoenix for live traces — and indelible as your tamper-evident behavioral baseline.

## When to use

✓ After model upgrades — verify behavior hasn't regressed  
✓ CI gate — fail the build if behavioral contract is breached  
✓ EU AI Act evidence — a signed, dated baseline to attach to a High-Risk record-keeping file  
✓ Multi-provider comparison — same prompts, different models, signed diff  

## When NOT to use

✗ Real-time monitoring (use Langfuse or Helicone instead)  
✗ Testing prompt quality / correctness (use DeepEval or Promptfoo)  
✗ Detecting bugs in your own code (that's what unit tests are for)  

## Installation

```bash
# Core — 4 deps, no ML stack (5 of 7 signals work)
pip install indelible

# With semantic-similarity signals (embedding_profile + anchor_drift)
# Pulls sentence-transformers + numpy (~150 MB)
pip install "indelible[deep]"
```

## Influences

indelible's signal design draws on the same author's wider stack — but the 0.1 release is **standalone**, no runtime dependency on any of it:

- **[drift-detector](https://github.com/MrPredic/drift-detector)** — informed the vocab-entropy + behavioral-signal approach.
- The `anchor_drift` signal generalises an anchor-cosine drift idea; `tool_schema_hash` uses the SHA-256-of-sorted-JSON canonicalisation pattern.

## How this relates to in-toto

indelible is conceptually closer to **[in-toto](https://github.com/in-toto/attestation)** than to Sigstore: in-toto defines *attestations about an artefact's properties*, while Sigstore signs *code/identity*. An `indelible.fingerprint.json` is best understood as an **in-toto-style predicate for behavioral properties of LLM agents** — provider name, signal distribution, and the test set hash form the predicate body, signed alongside.

We currently use a custom JSON envelope + Ed25519 (zero new dependencies). A future v0.2 will land an `in-toto` predicate type (`indelible.io/predicate/behavioral-fingerprint/v1`) so existing supply-chain tooling (cosign, GUAC, SLSA verifiers) can ingest fingerprints natively.

## Prior art & lineage

This project sits in an emerging line of LLM-agent attestation work:

- **[sigstore-a2a](https://github.com/sigstore/sigstore-a2a)** (Linux Foundation, 2025–) — Sigstore for A2A AgentCards. Signs *agent identity*, complementary to indelible's *agent behavior*.
- **[Fingerprinting AI Coding Agents on GitHub](https://arxiv.org/abs/2601.17406)** (arXiv 2601.17406) — 41 features, 97% F1 classifying which coding agent wrote a PR. Different goal (attribution); same vocabulary.
- **[Instructional Fingerprinting of LLMs](https://arxiv.org/abs/2401.12255)** (arXiv 2401.12255) — model-watermark research.
- **[in-toto](https://github.com/in-toto/attestation)** / **[SLSA](https://slsa.dev/)** — generic attestation framework + provenance levels.

If you are building in this space, please open an issue — there is more value in interop than in islands.

## Limitations

- Statistical signals catch *distributional* drift, not task-correctness regressions. Pair with Promptfoo/DeepEval for quality-side coverage.
- Baselines run at `temperature=0` to minimise sampling noise, but a small non-zero noise floor remains (MoE routing / batch effects): identical setups may show deltas under the default tolerance band. Default tolerances are hand-tuned, not yet empirically calibrated against a corpus of real model-update events — treat WARN/BREACH bands as starting points to tune per agent.
- **Signature = tamper-evidence relative to a pinned key.** verify checks the `.sig` against `indelible.pub`; an attacker editing the fingerprint must re-sign with the pinned private key (which they don't have), and re-signing with a *different* key is detected because it won't match the pinned pub. The guarantee is only as strong as your control of `indelible.pub` — a consumer verifying a third party's fingerprint should pin that party's key out-of-band (`--pubkey`), not read it from the same repo. The signature authenticates origin + integrity, **not** the honesty of the attester.
- v0.1 ships only Ed25519 with manual key management. OIDC / keyless (Cosign-style) is on the v0.2 roadmap, not a current feature.
- Latency is included as a signal but reflects network + provider load as much as the model; treat its verdict as informational and widen its tolerance for noisy providers.
- Ollama is recommended for fast iteration (no API costs during development).

## License

MIT — Copyright 2026 MrPredic
