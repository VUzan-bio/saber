"""Tests for heuristic scoring."""

import pytest

from saber.core.types import CrRNACandidate, OffTargetReport
from saber.scoring.heuristic import HeuristicScorer


class TestHeuristicScorer:
    def test_score_produces_composite(self, sample_candidate, clean_offtarget):
        scorer = HeuristicScorer()
        result = scorer.score(sample_candidate, clean_offtarget)
        assert 0.0 <= result.heuristic.composite <= 1.0

    def test_seed_position_1_is_best(self, sample_candidate, clean_offtarget):
        scorer = HeuristicScorer()

        # Position 1 (best)
        sample_candidate.mutation_position_in_spacer = 1
        score_1 = scorer.score(sample_candidate, clean_offtarget)

        # Position 8 (worst in seed)
        sample_candidate.mutation_position_in_spacer = 8
        score_8 = scorer.score(sample_candidate, clean_offtarget)

        assert score_1.heuristic.seed_position_score > score_8.heuristic.seed_position_score

    def test_batch_ranking(self, sample_candidate, clean_offtarget):
        scorer = HeuristicScorer()
        scored = scorer.score_batch(
            [sample_candidate, sample_candidate],
            [clean_offtarget, clean_offtarget],
        )
        assert scored[0].rank == 1
        assert scored[1].rank == 2

    def test_gc_50_is_optimal(self):
        score = HeuristicScorer._score_gc(0.50)
        assert score == 1.0

    def test_gc_penalty_symmetric(self):
        score_low = HeuristicScorer._score_gc(0.40)
        score_high = HeuristicScorer._score_gc(0.60)
        assert abs(score_low - score_high) < 0.01
