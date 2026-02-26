"""Track experimental validation results and close the active learning loop.

This module serves two purposes:
1. Bookkeeping: log which candidates were tested, under what conditions, with what results.
2. Active learning: compare predictions to measurements, identify where the model
   is most uncertain, and recommend the next batch of candidates to validate.

The active learning cycle:
    Cycle N: SABER designs + ranks candidates → validate top-K and bottom-K →
    retrain JEPA on new data → Cycle N+1: improved predictions.

Data is stored as JSON lines for simplicity and git-friendliness.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from saber.core.types import ExperimentalResult, ScoredCandidate

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Track experimental results and support active learning.

    Usage:
        tracker = ExperimentTracker(db_path="results/experiments.jsonl")
        tracker.log_result(result)
        summary = tracker.summary()
        next_batch = tracker.suggest_next_batch(scored_candidates, k=10)
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._results: Optional[list[ExperimentalResult]] = None

    def log_result(self, result: ExperimentalResult) -> None:
        """Append a new experimental result."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "a") as f:
            f.write(result.model_dump_json() + "\n")
        # Invalidate cache
        self._results = None
        logger.info("Logged result for candidate %s", result.candidate_id)

    def log_batch(self, results: list[ExperimentalResult]) -> None:
        """Append multiple results."""
        for r in results:
            self.log_result(r)

    def load_all(self) -> list[ExperimentalResult]:
        """Load all results from the database."""
        if self._results is not None:
            return self._results
        if not self.db_path.exists():
            return []
        results: list[ExperimentalResult] = []
        with open(self.db_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(ExperimentalResult.model_validate_json(line))
        self._results = results
        return results

    def get_tested_ids(self) -> set[str]:
        """Return set of candidate IDs that have been experimentally tested."""
        return {r.candidate_id for r in self.load_all()}

    def summary(self) -> dict:
        """Summary statistics of all logged experiments."""
        results = self.load_all()
        if not results:
            return {"total": 0}

        tested_ids = self.get_tested_ids()
        disc_ratios = [r.discrimination_ratio for r in results if r.discrimination_ratio is not None]

        return {
            "total_measurements": len(results),
            "unique_candidates": len(tested_ids),
            "assay_types": list({r.assay_type for r in results}),
            "mean_discrimination_ratio": (
                sum(disc_ratios) / len(disc_ratios) if disc_ratios else None
            ),
            "validated_count": len([r for r in results if r.discrimination_ratio and r.discrimination_ratio > 2.0]),
        }

    def prediction_vs_measurement(
        self,
        scored: list[ScoredCandidate],
    ) -> list[dict]:
        """Compare predicted scores with experimental measurements.

        Returns list of {candidate_id, predicted, measured, error} dicts.
        Used for evaluating model accuracy and identifying systematic biases.
        """
        results_by_id: dict[str, list[ExperimentalResult]] = {}
        for r in self.load_all():
            results_by_id.setdefault(r.candidate_id, []).append(r)

        comparisons: list[dict] = []
        for s in scored:
            cid = s.candidate.candidate_id
            if cid not in results_by_id:
                continue

            predicted = s.heuristic.composite
            if s.ml_scores:
                predicted = s.ml_scores[0].predicted_efficiency

            measurements = results_by_id[cid]
            mean_measured = sum(m.signal_value for m in measurements) / len(measurements)

            comparisons.append({
                "candidate_id": cid,
                "predicted": predicted,
                "measured": mean_measured,
                "error": abs(predicted - mean_measured),
                "n_measurements": len(measurements),
            })

        return comparisons

    def suggest_next_batch(
        self,
        scored: list[ScoredCandidate],
        k: int = 10,
        strategy: str = "balanced",
    ) -> list[ScoredCandidate]:
        """Suggest the next K candidates to experimentally validate.

        Strategies:
        - "top": pick the K highest-ranked untested candidates.
        - "uncertain": pick candidates where model confidence is lowest.
        - "balanced": pick top K/2 + bottom K/2 for maximum learning signal.

        The balanced strategy is most informative for active learning:
        testing both predicted-good and predicted-bad candidates provides
        the strongest training signal for the next JEPA iteration.
        """
        tested = self.get_tested_ids()
        untested = [s for s in scored if s.candidate.candidate_id not in tested]

        if not untested:
            logger.warning("All candidates have been tested")
            return []

        if strategy == "top":
            return untested[:k]

        elif strategy == "uncertain":
            # Sort by model uncertainty (if available)
            def uncertainty(s: ScoredCandidate) -> float:
                if s.ml_scores and s.ml_scores[0].confidence is not None:
                    return 1.0 - s.ml_scores[0].confidence
                return 0.5  # no uncertainty info → medium priority
            untested.sort(key=uncertainty, reverse=True)
            return untested[:k]

        elif strategy == "balanced":
            # Top half + bottom half
            half = k // 2
            top = untested[:half]
            bottom = untested[-half:] if len(untested) > half else []
            return top + bottom

        raise ValueError(f"Unknown strategy: {strategy}")

    def export_training_data(self, output_path: str | Path) -> int:
        """Export experimental results in a format suitable for JEPA fine-tuning.

        Returns the number of exported samples.
        """
        results = self.load_all()
        if not results:
            return 0

        training_data = []
        for r in results:
            training_data.append({
                "candidate_id": r.candidate_id,
                "signal_value": r.signal_value,
                "discrimination_ratio": r.discrimination_ratio,
                "assay_type": r.assay_type,
            })

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(training_data, f, indent=2)

        logger.info("Exported %d training samples to %s", len(training_data), output)
        return len(training_data)
