"""Embedding profile collector (requires [deep] extra)."""
from __future__ import annotations

import logging
from typing import List

from indelible.types import Signal

logger = logging.getLogger(__name__)


class EmbeddingProfileCollector:
    name = "embedding_profile"
    needs_extras: tuple = ("deep",)

    def collect(
        self,
        outputs: List[str],
        inputs: List[str],
        anchor_text: str,
        tools_called: List[List[str]],
    ) -> Signal:
        # tol 0.05 = ±0.05 cosine. Embedding cosines on the same outputs are
        # near-deterministic (variance ≈ 1e-6 for identical strings); 0.05
        # absorbs cross-run float noise without missing real semantic shifts.
        try:
            from sentence_transformers import SentenceTransformer, util  # type: ignore
        except ImportError:
            logger.warning("sentence-transformers/numpy not available; skipping %s", self.name)
            return Signal(name=self.name, value=0.0, tolerance=0.05)

        if not outputs:
            return Signal(name=self.name, value=0.0, tolerance=0.05)

        import numpy as np  # type: ignore  # noqa: PLC0415
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = np.array(model.encode(outputs))
        centroid = embeddings.mean(axis=0)
        sims = [float(util.cos_sim(centroid, e)) for e in embeddings]
        mean_sim = sum(sims) / len(sims)
        variance = sum((s - mean_sim) ** 2 for s in sims) / len(sims)
        return Signal(name=self.name, value=mean_sim, p50=variance, tolerance=0.05)
