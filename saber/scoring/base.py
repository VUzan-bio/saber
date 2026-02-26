"""Abstract scorer interface.

All scoring backends (heuristic, CNN, JEPA) implement this protocol.
The pipeline doesn't care which backend is active — it just calls .score().
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from saber.core.types import (
    CrRNACandidate,
    HeuristicScore,
    MLScore,
    OffTargetReport,
    ScoredCandidate,
)


class Scorer(ABC):
    """Base class for all scoring backends."""

    @abstractmethod
    def score(
        self,
        candidate: CrRNACandidate,
        offtarget: OffTargetReport,
    ) -> ScoredCandidate:
        """Score a single candidate."""
        ...

    def score_batch(
        self,
        candidates: list[CrRNACandidate],
        offtargets: list[OffTargetReport],
    ) -> list[ScoredCandidate]:
        """Score and rank a batch. Override for GPU-batched implementations."""
        assert len(candidates) == len(offtargets)
        scored = [self.score(c, o) for c, o in zip(candidates, offtargets)]
        scored.sort(key=lambda s: self._sort_key(s), reverse=True)
        for i, s in enumerate(scored):
            s.rank = i + 1
        return scored

    @staticmethod
    def _sort_key(s: ScoredCandidate) -> float:
        """Primary sort: ML score if available, else heuristic composite."""
        if s.ml_scores:
            return s.ml_scores[0].predicted_efficiency
        return s.heuristic.composite
