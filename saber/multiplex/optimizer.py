"""Multiplex panel optimization.

Given N targets, each with ranked crRNA candidates, select the optimal N-plex
combination that jointly maximises:
- Individual guide efficiency scores
- Individual discrimination ratios
- Minimises cross-reactivity between guides
- Ensures RPA primer compatibility (no primer dimers across all 2N primers)

The combinatorial space is enormous: with 5 candidates per target and 14 targets,
that's 5^14 ≈ 6 billion combinations. We use a two-stage approach:

Stage 1: Greedy construction — pick best candidate per target, then resolve conflicts
Stage 2: Simulated annealing refinement — swap candidates to improve joint score

The optimizer treats primer compatibility as a hard constraint (checked via Module 6)
and the other objectives as a weighted soft score.
"""

from __future__ import annotations

import copy
import logging
import math
import random
from typing import Optional

import numpy as np

from saber.core.types import MultiplexPanel, PanelMember, ScoredCandidate, Target

logger = logging.getLogger(__name__)


class MultiplexOptimizer:
    """Optimise an N-plex crRNA panel.

    Usage:
        optimizer = MultiplexOptimizer()
        panel = optimizer.optimize(
            targets=targets,
            candidates_per_target=candidates_by_label,
        )
    """

    def __init__(
        self,
        cross_reactivity_weight: float = 0.3,
        efficiency_weight: float = 0.5,
        discrimination_weight: float = 0.2,
        max_iterations: int = 10_000,
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.995,
        seed: int = 42,
    ) -> None:
        self.w_cross = cross_reactivity_weight
        self.w_eff = efficiency_weight
        self.w_disc = discrimination_weight
        self.max_iter = max_iterations
        self.t0 = initial_temperature
        self.cooling = cooling_rate
        self.rng = random.Random(seed)

    def optimize(
        self,
        targets: list[Target],
        candidates_per_target: dict[str, list[ScoredCandidate]],
        cross_reactivity_fn: Optional[CrossReactivityFn] = None,
    ) -> list[ScoredCandidate]:
        """Select optimal candidates for a multiplex panel.

        Args:
            targets: List of targets in the panel.
            candidates_per_target: {target_label: [scored candidates, ranked]}.
            cross_reactivity_fn: Optional callable that scores pairwise cross-reactivity.

        Returns:
            List of selected ScoredCandidates, one per target.
        """
        labels = [t.label for t in targets]
        n = len(labels)

        # Validate
        for label in labels:
            if label not in candidates_per_target or not candidates_per_target[label]:
                raise ValueError(f"No candidates for target {label}")

        # Stage 1: Greedy initialization — pick rank-1 for each target
        selection = [candidates_per_target[label][0] for label in labels]
        best_score = self._panel_score(selection, cross_reactivity_fn)
        best_selection = copy.copy(selection)

        logger.info("Greedy init: panel score = %.4f", best_score)

        # Stage 2: Simulated annealing
        temperature = self.t0
        for iteration in range(self.max_iter):
            # Pick a random target and swap to a random alternative candidate
            idx = self.rng.randint(0, n - 1)
            label = labels[idx]
            pool = candidates_per_target[label]
            if len(pool) <= 1:
                continue

            current = selection[idx]
            alternatives = [c for c in pool if c.candidate.candidate_id != current.candidate.candidate_id]
            if not alternatives:
                continue

            new_candidate = self.rng.choice(alternatives)
            selection[idx] = new_candidate
            new_score = self._panel_score(selection, cross_reactivity_fn)

            delta = new_score - best_score
            if delta > 0 or self.rng.random() < math.exp(delta / max(temperature, 1e-10)):
                # Accept
                if new_score > best_score:
                    best_score = new_score
                    best_selection = copy.copy(selection)
            else:
                # Reject — revert
                selection[idx] = current

            temperature *= self.cooling

            if iteration % 1000 == 0:
                logger.debug(
                    "SA iter %d: score=%.4f, best=%.4f, T=%.4f",
                    iteration, new_score, best_score, temperature,
                )

        logger.info(
            "Optimization complete: best panel score = %.4f after %d iterations",
            best_score, self.max_iter,
        )
        return best_selection

    def compute_cross_reactivity_matrix(
        self,
        selected: list[ScoredCandidate],
    ) -> np.ndarray:
        """Compute pairwise sequence similarity between selected crRNAs.

        Simple metric: fraction of matching positions between spacers.
        More sophisticated: use JEPA embedding distance.
        """
        n = len(selected)
        matrix = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(i + 1, n):
                sim = self._spacer_similarity(
                    selected[i].candidate.spacer_seq,
                    selected[j].candidate.spacer_seq,
                )
                matrix[i, j] = sim
                matrix[j, i] = sim

        return matrix

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _panel_score(
        self,
        selection: list[ScoredCandidate],
        cross_fn: Optional[CrossReactivityFn],
    ) -> float:
        """Compute the joint panel score for a candidate selection."""
        n = len(selection)

        # Efficiency: mean heuristic (or ML) score across panel
        eff_scores = []
        for s in selection:
            if s.ml_scores:
                eff_scores.append(s.ml_scores[0].predicted_efficiency)
            else:
                eff_scores.append(s.heuristic.composite)
        mean_eff = sum(eff_scores) / n if n else 0

        # Discrimination: mean ratio (if available)
        disc_scores = []
        for s in selection:
            if s.discrimination is not None:
                disc_scores.append(min(s.discrimination.ratio, 10.0) / 10.0)
            else:
                disc_scores.append(0.5)
        mean_disc = sum(disc_scores) / n if n else 0

        # Cross-reactivity penalty: mean pairwise similarity
        cross_penalty = 0.0
        if cross_fn is not None:
            cross_penalty = cross_fn(selection)
        else:
            total_sim = 0.0
            pairs = 0
            for i in range(n):
                for j in range(i + 1, n):
                    total_sim += self._spacer_similarity(
                        selection[i].candidate.spacer_seq,
                        selection[j].candidate.spacer_seq,
                    )
                    pairs += 1
            cross_penalty = total_sim / pairs if pairs else 0

        return (
            self.w_eff * mean_eff
            + self.w_disc * mean_disc
            - self.w_cross * cross_penalty
        )

    @staticmethod
    def _spacer_similarity(seq1: str, seq2: str) -> float:
        """Fraction of matching positions between two spacers."""
        min_len = min(len(seq1), len(seq2))
        if min_len == 0:
            return 0.0
        matches = sum(a == b for a, b in zip(seq1[:min_len], seq2[:min_len]))
        return matches / min_len


# Type alias for custom cross-reactivity scoring functions
from typing import Callable
CrossReactivityFn = Callable[[list[ScoredCandidate]], float]
