"""Level 1 — Rule-based heuristic scoring.

Based on feature importance analysis from Kim et al. (2018) and
empirical rules from Cas12a guide design literature.

Each sub-score is normalised to [0, 1] where 1 = optimal.
The composite score is a weighted sum.

This is the baseline that works immediately without any training data.
"""

from __future__ import annotations

import math

from saber.core.constants import (
    GC_MAX,
    GC_MIN,
    HEURISTIC_WEIGHTS,
    HOMOPOLYMER_MAX,
    MFE_THRESHOLD,
    SEED_REGION_END,
)
from saber.core.types import (
    CrRNACandidate,
    HeuristicScore,
    OffTargetReport,
    ScoredCandidate,
)
from saber.scoring.base import Scorer


class HeuristicScorer(Scorer):
    """Rule-based crRNA scoring.

    Usage:
        scorer = HeuristicScorer()
        scored = scorer.score_batch(candidates, offtargets)
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or HEURISTIC_WEIGHTS

    def score(
        self,
        candidate: CrRNACandidate,
        offtarget: OffTargetReport,
    ) -> ScoredCandidate:
        seed_score = self._score_seed_position(candidate.mutation_position_in_spacer)
        gc_penalty = self._score_gc(candidate.gc_content)
        structure_penalty = self._score_structure(candidate.mfe)
        homo_penalty = self._score_homopolymer(candidate.homopolymer_max)
        ot_penalty = self._score_offtarget(offtarget)

        composite = (
            self.weights["seed_position"] * seed_score
            + self.weights["gc"] * gc_penalty
            + self.weights["structure"] * structure_penalty
            + self.weights["homopolymer"] * homo_penalty
            + self.weights["offtarget"] * ot_penalty
        )

        heuristic = HeuristicScore(
            seed_position_score=seed_score,
            gc_penalty=gc_penalty,
            structure_penalty=structure_penalty,
            homopolymer_penalty=homo_penalty,
            offtarget_penalty=ot_penalty,
            composite=composite,
        )

        return ScoredCandidate(
            candidate=candidate,
            offtarget=offtarget,
            heuristic=heuristic,
        )

    # ------------------------------------------------------------------
    # Sub-scores, each normalised to [0, 1]
    # ------------------------------------------------------------------

    @staticmethod
    def _score_seed_position(pos: int) -> float:
        """Closer to PAM = better discrimination. Linear decay from pos 1-8."""
        if pos > SEED_REGION_END:
            return 0.0
        return 1.0 - (pos - 1) / SEED_REGION_END

    @staticmethod
    def _score_gc(gc: float) -> float:
        """Optimal GC is 50% for Cas12a. Penalise deviation."""
        optimal = 0.50
        max_deviation = max(abs(GC_MAX - optimal), abs(GC_MIN - optimal))
        deviation = abs(gc - optimal)
        return max(0.0, 1.0 - deviation / max_deviation)

    @staticmethod
    def _score_structure(mfe: float | None) -> float:
        """Less negative MFE = less secondary structure = better.

        MFE of 0 (no structure) → score 1.0
        MFE at threshold → score 0.0
        """
        if mfe is None:
            return 0.5  # no data → neutral score
        if mfe >= 0:
            return 1.0
        return max(0.0, 1.0 - mfe / MFE_THRESHOLD)

    @staticmethod
    def _score_homopolymer(max_run: int) -> float:
        """No homopolymers → 1.0. At max → 0.0."""
        if max_run <= 1:
            return 1.0
        return max(0.0, 1.0 - (max_run - 1) / HOMOPOLYMER_MAX)

    @staticmethod
    def _score_offtarget(report: OffTargetReport) -> float:
        """Clean (no risky hits) → 1.0. Exponential decay with hit count."""
        n = report.total_risky_hits
        if n == 0:
            return 1.0
        return math.exp(-0.5 * n)
