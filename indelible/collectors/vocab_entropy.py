"""Vocabulary Shannon-entropy collector."""
from __future__ import annotations

import math
from collections import Counter
from typing import List

from indelible.types import Signal


class VocabEntropyCollector:
    name = "vocab_entropy"
    needs_extras: tuple = ()

    def collect(
        self,
        outputs: List[str],
        inputs: List[str],
        anchor_text: str,
        tools_called: List[List[str]],
    ) -> Signal:
        # tol 0.50 bits absolute. Output distributions wobble naturally across
        # 0.1-0.3 bits run-to-run; a >0.5 bit shift means the vocabulary
        # actually drifted (terser/more-verbose model, different sampling temp).
        tokens = " ".join(outputs).lower().split()
        if not tokens:
            return Signal(name=self.name, value=0.0, tolerance=0.50)
        counts = Counter(tokens)
        total = len(tokens)
        entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
        return Signal(name=self.name, value=entropy, tolerance=0.50)
