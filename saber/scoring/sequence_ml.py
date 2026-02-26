"""Level 2 — Sequence-based ML prediction (Seq-deepCpf1 equivalent).

CNN or small transformer trained directly on Kim et al. 2018 HT-PAMDA data.
Input: one-hot encoded 34-nt context (4 PAM + 20 spacer + 10 flanking).
Output: predicted indel frequency / cleavage rate.

This serves as the baseline ML model that JEPA must outperform.

Reference: Kim et al., Nature Biotechnology (2018). "Deep learning improves
prediction of CRISPR–Cpf1 guide RNA activity."
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from saber.core.types import (
    CrRNACandidate,
    HeuristicScore,
    MLScore,
    OffTargetReport,
    ScoredCandidate,
)
from saber.scoring.base import Scorer

logger = logging.getLogger(__name__)

# One-hot encoding: A=0, T=1, G=2, C=3
_NT_TO_IDX = {"A": 0, "T": 1, "G": 2, "C": 3}
_CONTEXT_LENGTH = 34  # 4 (PAM) + 20 (spacer) + 10 (flanking)


def one_hot_encode(seq: str) -> np.ndarray:
    """Encode a nucleotide sequence as a (4, L) one-hot matrix."""
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, nt in enumerate(seq.upper()):
        idx = _NT_TO_IDX.get(nt)
        if idx is not None:
            arr[idx, i] = 1.0
    return arr


class SequenceMLScorer(Scorer):
    """Sequence-based CNN scorer.

    Wraps a trained PyTorch model. Falls back to heuristic if model
    is unavailable.

    Usage:
        scorer = SequenceMLScorer(model_path="checkpoints/seq_cnn.pt")
        scored = scorer.score_batch(candidates, offtargets)
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        heuristic_fallback: Optional[Scorer] = None,
    ) -> None:
        self.model = None
        self.model_path = model_path
        self._fallback = heuristic_fallback

        if model_path is not None:
            self._load_model(Path(model_path))

    def score(
        self,
        candidate: CrRNACandidate,
        offtarget: OffTargetReport,
    ) -> ScoredCandidate:
        # Always compute heuristic as baseline
        if self._fallback:
            base = self._fallback.score(candidate, offtarget)
        else:
            from saber.scoring.heuristic import HeuristicScorer
            base = HeuristicScorer().score(candidate, offtarget)

        # Add ML prediction if model available
        if self.model is not None:
            prediction = self._predict(candidate)
            base.ml_scores.append(MLScore(
                model_name="seq_cnn",
                predicted_efficiency=prediction,
            ))

        return base

    def score_batch(
        self,
        candidates: list[CrRNACandidate],
        offtargets: list[OffTargetReport],
    ) -> list[ScoredCandidate]:
        """Override for GPU-batched inference."""
        if self.model is None:
            return super().score_batch(candidates, offtargets)

        # Batch encode
        contexts = [self._encode_context(c) for c in candidates]
        predictions = self._predict_batch(contexts)

        scored = []
        for c, o, pred in zip(candidates, offtargets, predictions):
            s = self.score(c, o)
            # Replace the individual prediction with batch result
            s.ml_scores = [MLScore(model_name="seq_cnn", predicted_efficiency=pred)]
            scored.append(s)

        scored.sort(key=lambda s: self._sort_key(s), reverse=True)
        for i, s in enumerate(scored):
            s.rank = i + 1
        return scored

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_model(self, path: Path) -> None:
        """Load a trained PyTorch model."""
        try:
            import torch
            self.model = torch.jit.load(str(path), map_location="cpu")
            self.model.eval()
            logger.info("Loaded sequence ML model from %s", path)
        except Exception as e:
            logger.warning("Failed to load model from %s: %s", path, e)
            self.model = None

    def _encode_context(self, candidate: CrRNACandidate) -> np.ndarray:
        """Build the 34-nt context: PAM (4) + spacer (20) + flanking (10)."""
        context = candidate.pam_seq + candidate.spacer_seq
        # Pad or truncate to CONTEXT_LENGTH
        if len(context) < _CONTEXT_LENGTH:
            context = context + "N" * (_CONTEXT_LENGTH - len(context))
        else:
            context = context[:_CONTEXT_LENGTH]
        return one_hot_encode(context)

    def _predict(self, candidate: CrRNACandidate) -> float:
        """Single-sample prediction."""
        encoded = self._encode_context(candidate)
        predictions = self._predict_batch([encoded])
        return predictions[0]

    def _predict_batch(self, contexts: list[np.ndarray]) -> list[float]:
        """Batch prediction. Returns list of efficiency scores in [0, 1]."""
        if self.model is None:
            return [0.5] * len(contexts)

        import torch
        batch = torch.tensor(np.stack(contexts), dtype=torch.float32)
        with torch.no_grad():
            output = self.model(batch)
        return output.squeeze(-1).clamp(0, 1).tolist()
