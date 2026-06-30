"""Generic OpenAI-compatible provider (Groq, Together, vLLM, etc.)."""
from __future__ import annotations

import copy
import time
from typing import List, Optional, Tuple

import httpx

from indelible.providers.errors import raise_for_status


class OpenAICompatProvider:
    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def complete(
        self, system: str, user: str, tools: Optional[list] = None
    ) -> Tuple[str, List[str], float]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        body: dict = {"model": self.model, "messages": messages}
        if tools:
            body["tools"] = copy.deepcopy(tools)

        headers: dict = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        t0 = time.perf_counter()
        resp = httpx.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=60)
        latency = time.perf_counter() - t0

        raise_for_status(resp.status_code, resp.text)

        data = resp.json()
        message = data["choices"][0]["message"]
        raw = message.get("content")
        content: str = raw if raw is not None else ""

        tool_calls = message.get("tool_calls") or []
        tools_called = [tc["function"]["name"] for tc in tool_calls if "function" in tc]

        return content, tools_called, latency
