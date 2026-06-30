"""Local Ollama provider."""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import httpx

from indelible.providers.errors import raise_for_status


class OllamaProvider:
    def __init__(self, model: str, host: str = "http://localhost:11434") -> None:
        self.model = model.removeprefix("ollama/")
        self.host = host.rstrip("/")

    def complete(
        self, system: str, user: str, tools: Optional[list] = None
    ) -> Tuple[str, List[str], float]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        body = {"model": self.model, "messages": messages, "stream": False}

        t0 = time.perf_counter()
        resp = httpx.post(f"{self.host}/api/chat", json=body, timeout=120)
        latency = time.perf_counter() - t0

        raise_for_status(resp.status_code, resp.text)

        content: str = resp.json()["message"]["content"]
        return content, [], latency
