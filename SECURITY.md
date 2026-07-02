# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | ✓ |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Email: security report via GitHub private vulnerability reporting at  
`https://github.com/MrPredic/indelible/security/advisories/new`

We aim to respond within 72 hours and release a patch within 14 days for confirmed issues.

## Trust model

The signature provides **tamper-evidence relative to a pinned public key** — it is not self-authenticating.

- `indelible init` writes the private key to `~/.indelible/key.pem` (kept out of the repo) and the public key to `indelible.pub` (committed).
- `indelible.pub` is the **trust anchor**. `verify` checks the `.sig` against it (or against `--pubkey`). The signature and the pinned key are deliberately kept apart: a public key that ships next to the signature is worthless, because an attacker who edits the fingerprint can re-sign with their own key and attach their own pub.
- Rotating the pinned key is therefore a **visible, reviewable event** (a `git diff` on `indelible.pub`, or a re-pin in the consumer's out-of-band store).
- **Cross-party verification:** a consumer verifying a third party's fingerprint MUST obtain that party's public key out-of-band (`--pubkey vendor.pub`) — never from the same repository that carries the fingerprint, or the anchor and the artefact share a fate.
- **What it does NOT guarantee:** authenticity + integrity of origin, not the *honesty* of the attester. A malicious attester can sign a truthful-looking-but-misleading baseline. indelible proves *who* produced the fingerprint and that it was not altered after signing — not that the baseline is meaningful.

## Scope

- Ed25519 key handling (`indelible/attest.py`, `indelible/verify.py`)
- Fingerprint signature verification bypass
- Path traversal in `IndelibleConfig.from_toml()` or CLI file arguments
- Dependency vulnerabilities in `cryptography >= 42`

## Out of scope

- Issues in optional extras (`[deep]`)
- Prompt injection in the models you are attesting (indelible is a measurement tool, not a firewall)
