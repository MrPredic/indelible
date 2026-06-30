# Changelog

## [Unreleased] — Roadmap

### v0.2 — Adoption accelerators
- GitHub Action `indelible-verify` as reusable CI workflow
- in-toto predicate type (`indelible.io/predicate/behavioral-fingerprint/v1`)
- OIDC / keyless signing (Cosign-style)
- Retry helper around `provider.complete()` (typed-exception aware)
- Output-length distribution signal (cheapest catch for "model got terser")

### v0.3 — Vertical templates
- `indelible init --template coding` (50 SWE test inputs)
- Customer-support + research-assistant templates
- Hosted "verify hub" (optional): publish fingerprints publicly

### v0.4 — Compliance bundle
- EU AI Act audit report generator (PDF export)
- HIPAA / SOC2 / FINRA templates

### v1.0 — Standardisation
- Fingerprint format RFC submission
- ETDI / A2A spec integration
- ZK-Proof extension (privacy-preserving verification)

---

## [0.1.0] — Pre-release (review-hardened)

> **Renamed before first publish:** this project was developed under the working
> name `bedrock-attest`. It was renamed to **`indelible`** prior to release to
> avoid collision with Amazon Bedrock (which owns the `bedrock-*` namespace on
> PyPI and dominates that search term). No `bedrock-attest` version was ever
> published to PyPI. The name *indelible* reflects the core property: a signed
> behavioral baseline you cannot quietly rewrite.

This is the post-pre-release-review build. All findings from a 5-track external
review (technical, architecture, strategic, marketing/AEO, competitive) have
been addressed at the P0/P1 level.

### Added
- `indelible init` — generates Ed25519 signing key + scaffolds `indelible.toml` / `prompts.json`
- `indelible attest` — runs test suite, collects 7 behavioral signals, writes signed fingerprint
- `indelible verify` — re-attests and compares tolerance-aware, colored report, exit codes
- `indelible diff` — side-by-side fingerprint comparison without re-attesting
- `indelible attest --config / --out / --prompts` and `indelible verify --config / --fp / --prompts` for multi-agent repos
- Signal collectors: `refusal_rate`, `latency` (P50/P95), `vocab_entropy`, `tool_distribution`, `tool_schema_hash`, `embedding_profile` (opt), `anchor_drift` (opt)
- Provider adapters: OpenAI-compatible (Groq, Together, vLLM, …), Anthropic, Ollama
- **Typed provider exceptions**: `ProviderError`, `ProviderAuthError` (401/403), `ProviderRateLimitError` (429), `ProviderServerError` (5xx) — all `RuntimeError` subclasses for backward compat
- `Signal.digest` — optional exact-match field for SHA-style signals (used by `tool_schema_hash`); verify does bytewise comparison instead of numeric tolerance
- `Fingerprint.canonical_bytes()` / `canonical_digest()` — single source of truth for sign + verify payload
- `IndelibleConfig.maintainer` — read from `[agent].maintainer` in TOML, signed into every fingerprint, **excluded from canonical_hash** (identifies who attested, not what)
- `IndelibleConfig.refusal_patterns` — `[refusal] patterns = [...]` overrides built-in EN+DE patterns; **included in canonical_hash** (different patterns = different baseline)
- Per-signal tolerance defaults documented in `docs/TOLERANCES.md`: `refusal=0.10`, `latency=0.30` (300ms), `vocab_entropy=0.50` bits, `tool_distribution=0.10`, `tool_schema_hash=0.0` (digest), `embedding/anchor=0.05`
- Smart `_api_key()` routing — picks the right env var based on `provider_url` host (Groq URL → `GROQ_API_KEY`, etc.) instead of iteration order
- Verify warns on signals present in fresh run but missing from saved fingerprint (schema-drift visibility)
- Verify short-circuits on `config_hash` / `test_set_hash` mismatch — saves provider tokens for runs whose verdict is already breach
- `[deep]` optional extra for embedding-based signals (sentence-transformers + numpy)

### Fixed
- **`tool_schema_hash` collision risk** — was numerically reducing SHA-256 to 8 hex digits, allowing tool-schema changes within tolerance × 1e9 to false-pass. Now uses `Signal.digest` exact match.
- **Signature artefact location** — `.sig` + companion `.sig.pub` now travel with the fingerprint (`indelible.fingerprint.json.sig`), not next to the private key. Survives `git pull` into CI. Backward-compat fallback for legacy paths.
- **Hard-coded `maintainer=""`** — was always emitting empty maintainer despite the field being part of the signed payload.
- **Anthropic multi-text block** — was only extracting the first text block; now concatenates all text blocks (Anthropic responses can interleave text/tool_use/text).
- **Signal int vs float byte stability** — `Signal(value=5)` and `Signal(value=5.0)` produced different signing JSON; now coerced via `__post_init__`.
- **OpenAI-compat `content=null`** — no longer crashes; coerces to empty string.
- **OpenAI-compat `content="0"`** — no longer coerced to empty string (falsy-but-non-empty regression).
- **`verify()` ignoring config_hash + test_set_hash** — verify could pass with a different config or different prompts; now both are explicit breach conditions.

### Changed
- README leads with the user pain ("AI agent silently drifts"), not the now-occupied "Sigstore for AI agents" framing.
- README includes a positioning matrix replacing the over-claiming Comparison table.
- README cites prior art: arXiv 2601.17406, 2401.12255, sigstore-a2a, in-toto, SLSA.
- `llms.txt` expanded from 63 → 247 lines: FAQ, per-signal tolerance table, provider quickstarts, exception types, prior-art lineage.

### Removed
- **`tiktoken`** dependency — was declared but never imported. Saves ~30 MB per wheel install.
- **`[drift]` extra** — claimed integration with `drift-detector-agent` was never implemented in code. Promising-but-not-shipping is worse than not promising.
- **`[cosign]` extra** — `sigstore` lib was declared but never imported. Returning in v0.2 with an actual implementation.
- **`[all]`** simplified to just `[deep]` (the only real extra).

### Tested
- 147 tests, 1 skipped, 97 % line coverage
- Ruff clean
- mypy clean (project config)
- `python -m build` + `twine check` PASSED (wheel + sdist)
- E2E smoke: init → attest (signs) → verify (checks signature + context guards) — all green
- CI: Python 3.9–3.12 matrix
