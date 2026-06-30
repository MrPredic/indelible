"""Refusal rate collector."""
from __future__ import annotations

import re
from typing import List, Optional, Sequence

from indelible.types import Signal

# Default English + German patterns. Multilingual agents should override
# via [refusal] patterns = [...] in indelible.toml — see docs/TOLERANCES.md.
DEFAULT_PATTERNS: tuple = (
    r"I cannot",
    r"I'm not able to",
    r"I can't help",
    r"I apologize, but I cannot",
    r"I apologize but I cannot",
    r"I'm unable to",
    r"verboten",
    r"darf ich nicht",
    r"kann ich nicht",
)


class RefusalCollector:
    name = "refusal_rate"
    needs_extras: tuple = ()

    def __init__(self, patterns: Optional[Sequence[str]] = None) -> None:
        # Distinguish "explicit empty list (error)" from "None means use defaults".
        if patterns is None:
            chosen = DEFAULT_PATTERNS
        elif len(patterns) == 0:
            raise ValueError("RefusalCollector: patterns must be non-empty")
        else:
            chosen = tuple(patterns)
        self._re = re.compile("|".join(chosen), re.IGNORECASE)

    def collect(
        self,
        outputs: List[str],
        inputs: List[str],
        anchor_text: str,
        tools_called: List[List[str]],
    ) -> Signal:
        # tol 0.10 = ±10pp. Refusal rate is naturally noisy run-to-run,
        # but a >10pp shift on the same prompt set is the "model got more
        # cautious" signal we actually care about.
        if not outputs:
            return Signal(name=self.name, value=0.0, tolerance=0.10)
        count = sum(1 for o in outputs if self._re.search(o))
        return Signal(name=self.name, value=count / len(outputs), tolerance=0.10)
