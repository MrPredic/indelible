"""Provider adapters — auto-detect from model string."""
from __future__ import annotations

import os
from typing import Optional, Protocol

from bedrock_attest.providers.anthropic import AnthropicProvider
from bedrock_attest.providers.ollama import OllamaProvider
from bedrock_attest.providers.openai_compat import OpenAICompatProvider

__all__ = ["Provider", "get_provider", "OpenAICompatProvider", "AnthropicProvider", "OllamaProvider"]


class Provider(Protocol):
    def complete(
        self, system: str, user: str, tools: Optional[list] = None
    ) -> tuple:  # (output: str, tools_called: list[str], latency_s: float)
        ...


def get_provider(model: str, provider_url: str, api_key: Optional[str] = None) -> Provider:
    """Return the correct adapter based on model string prefix."""
    if model.startswith("claude-") or model.startswith("anthropic/"):
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        return AnthropicProvider(model=model, api_key=api_key)
    if model.startswith("ollama/"):
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        return OllamaProvider(model=model, host=host)
    return OpenAICompatProvider(base_url=provider_url, model=model, api_key=api_key)
