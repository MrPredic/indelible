# Tolerance defaults

`Signal.tolerance` controls how strict `verify` is for each signal. The verdict logic in `indelible.verify._compare_signal`:

```
delta = |fresh.value − original.value|
delta ≤ tolerance × 0.5   → pass
delta ≤ tolerance         → warn
delta >  tolerance        → breach
```

The dataclass default is `0.05`, but every built-in collector sets its own — chosen empirically so identical agents produce `pass` and only meaningful drift produces `warn`/`breach`.

## Per-signal defaults

| Signal | Default | Unit | Why this number |
|---|---:|---|---|
| `refusal_rate` | **0.10** | fraction of outputs | A 10pp shift on the same prompt set is the "model got more cautious" signal. Tighter defaults trip on natural run-to-run variance. |
| `latency` | **0.30** | seconds (absolute) | 0.05s breach-storms in CI for slow providers — Anthropic p95 ≈ 2-4s, +50ms is 1.6% relative but 100% over a 0.05 tolerance. 300ms absorbs provider variance, catches real regressions. |
| `vocab_entropy` | **0.50** | bits (absolute) | Output distributions wobble 0.1-0.3 bits run-to-run. >0.5 bit shift means the vocabulary actually drifted (terser/more-verbose model, sampling temp change). |
| `tool_distribution` | **0.10** | distinct-tool count | `value` is "how many different tools the agent reached for". A delta of one tool is meaningful behavior change. |
| `tool_schema_hash` | **0.00** | exact match | Uses `Signal.digest` (SHA-256). Any bit-flip on the canonical tool spec is a breach — no tolerance band. |
| `embedding_profile` | **0.05** | cosine | Embedding cosines on identical strings are near-deterministic; 0.05 absorbs cross-run float noise. |
| `anchor_drift` | **0.05** | cosine distance | Same rationale as embedding_profile. |

## When to override

Set `tolerance_default` in `indelible.toml` if you want a global override (applied as a fallback when a Signal is constructed without an explicit tolerance):

```toml
[agent]
tolerance_default = 0.10
```

For per-signal overrides, build the `Signal` yourself and pass it to a custom collector — the v0.1 API does not support per-signal config keys (planned for v0.2).

## Verdict band logic

The `× 0.5` "pass band" is intentional: signals at exactly the tolerance edge are reported as `warn`, not `pass`. This gives one CI-cycle of grace before a breach lands, which matches how teams actually triage drift (notice → investigate → ship fix or relax tolerance).
