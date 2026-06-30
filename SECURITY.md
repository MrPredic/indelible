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

## Scope

- Ed25519 key handling (`indelible/attest.py`, `indelible/verify.py`)
- Fingerprint signature verification bypass
- Path traversal in `IndelibleConfig.from_toml()` or CLI file arguments
- Dependency vulnerabilities in `cryptography >= 42`

## Out of scope

- Issues in optional extras (`[deep]`)
- Prompt injection in the models you are attesting (indelible is a measurement tool, not a firewall)
