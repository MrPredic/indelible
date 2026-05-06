"""Anthropic Messages API provider (httpx, no SDK)."""
from __future__ import annotations

import copy
import time
from typing import List, Optional, Tuple

import httpx


class AnthropicProvider:
    API_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self.api_key = api_key

    def complete(
        self, system: str, user: str, tools: Optional[list] = None
    ) -> Tuple[str, List[str], float]:
        body: dict = {
            "model": self.model,
            "max_tokens": 1024,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if tools:
            body["tools"] = copy.deepcopy(tools)

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json",
        }

        t0 = time.perf_counter()
        resp = httpx.post(self.API_URL, json=body, headers=headers, timeout=60)
        latency = time.perf_counter() - t0

        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        blocks = data.get("content", [])

        # extract text from the first text block (not necessarily index 0)
        text_blocks = [b["text"] for b in blocks if b.get("type") == "text"]
        content = text_blocks[0] if text_blocks else ""

        tools_called = [b["name"] for b in blocks if b.get("type") == "tool_use"]

        return content, tools_called, latency
