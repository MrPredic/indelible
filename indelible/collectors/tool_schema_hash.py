"""Tool schema hash collector."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

from indelible.types import Signal


class ToolSchemaHashCollector:
    name = "tool_schema_hash"
    needs_extras: tuple = ()

    def __init__(self, tools: List[Dict[str, Any]]) -> None:
        canonical = json.dumps(tools, sort_keys=True, ensure_ascii=False)
        self._hash = hashlib.sha256(canonical.encode()).hexdigest()
        self._tool_count = len(tools)

    @property
    def schema_hash_str(self) -> str:
        return self._hash

    def collect(
        self,
        outputs: List[str],
        inputs: List[str],
        anchor_text: str,
        tools_called: List[List[str]],
    ) -> Signal:
        # value/tolerance kept for backward-compat plotting; verify uses
        # `digest` (exact hex match) — any tool-schema delta is a breach.
        return Signal(
            name=self.name,
            value=0.0,
            tolerance=0.0,
            distribution={"tool_count": float(self._tool_count)},
            digest=self._hash,
        )
