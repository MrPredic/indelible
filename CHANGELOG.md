# Changelog

## [Unreleased] — Roadmap

### v0.2 — Adoption accelerators
- GitHub Action `bedrock-verify` as reusable CI workflow
- Hosted "verify hub" (optional): publish fingerprints publicly
- Live demo on Hugging Face Spaces

### v0.3 — Vertical templates
- `bedrock init --template coding` (50 SWE test inputs)
- Customer-support + research-assistant templates

### v0.4 — Compliance bundle
- EU AI Act audit report generator (PDF export)
- HIPAA / SOC2 / FINRA templates

### v1.0 — Standardisation
- Fingerprint format RFC submission
- ETDI / A2A spec integration
- ZK-Proof extension (privacy-preserving verification)

---

## [0.1.0] — 2026-05-06

### Added
- `bedrock init` — generates Ed25519 signing key + scaffolds `bedrock.toml` / `prompts.json`
- `bedrock attest` — runs test suite, collects 7 behavioral signals, writes signed fingerprint
- `bedrock verify` — re-attests and compares tolerance-aware, colored report, exit codes
- `bedrock diff` — side-by-side fingerprint comparison without re-attesting
- Signal collectors: `refusal_rate`, `latency` (P50/P95), `vocab_entropy`, `tool_distribution`, `tool_schema_hash`, `embedding_profile` (opt), `anchor_drift` (opt)
- Provider adapters: OpenAI-compatible (Groq, Together, vLLM, …), Anthropic, Ollama
- Ed25519 signing via `cryptography` lib; optional Cosign (`[cosign]` extra)
- `BedrockConfig.from_toml()` / `to_toml()` round-trip
- `Fingerprint.to_dict()` / `from_dict()` — JSON-serialisable
- `[deep]` optional extra for embedding-based signals (sentence-transformers)
- `[drift]` optional extra for drift-detector-agent integration
- CI: Python 3.9–3.12 matrix, ruff + pytest
- 99 tests, 97 % line coverage
