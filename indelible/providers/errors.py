"""Typed provider exceptions — additive over RuntimeError."""
from __future__ import annotations


class ProviderError(RuntimeError):
    """Base for all provider-side errors. Carries status_code + body snippet.

    Subclasses of RuntimeError so existing ``except RuntimeError`` users still
    catch them — additive change, not breaking. Callers wanting retry logic
    can now branch on the specific subclass.
    """

    def __init__(self, status_code: int, body: str) -> None:
        snippet = body[:200] if body else ""
        super().__init__(f"HTTP {status_code}: {snippet}")
        self.status_code = status_code
        self.body = snippet


class ProviderAuthError(ProviderError):
    """401/403 — bad/missing API key. Do not retry."""


class ProviderRateLimitError(ProviderError):
    """429 — rate-limited. Retry with backoff."""


class ProviderServerError(ProviderError):
    """5xx — provider-side outage. Retry with backoff."""


def raise_for_status(status_code: int, body: str) -> None:
    """Raise the typed exception that matches the HTTP status."""
    if status_code == 200:
        return
    if status_code in (401, 403):
        raise ProviderAuthError(status_code, body)
    if status_code == 429:
        raise ProviderRateLimitError(status_code, body)
    if 500 <= status_code < 600:
        raise ProviderServerError(status_code, body)
    raise ProviderError(status_code, body)
