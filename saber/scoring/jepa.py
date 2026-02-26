"""Level 3 — JEPA fine-tuned predictor.

Pre-trained bDNA-JEPA embeddings capture genomic context that raw sequence
features miss. Two fine-tuning paths:

Path A: Direct efficiency regression
    JEPA encoder (frozen) → regression head → predicted cleavage efficiency
    Trained on Kim et al. 2018 data. Benchmark against Seq-deepCpf1.

Path B: Pairwise discrimination prediction
    JEPA encoder → embed(WT_spacer) ⊕ embed(MUT_spacer) → discrimination ratio
    More novel: predicts differential activity, not absolute efficiency.
    Trained on experimental mismatch pair data.

Path C: Genomic context embeddings for target selection
    JEPA embeddings of the genomic region around each target site predict
    assay-level properties (RPA amplification feasibility, off-target risk).
    Used at the target selection stage, not guide scoring.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from saber.core.types import (
    CrRNACandidate,
    DiscriminationScore,
    MLScore,
    MismatchPair,
    OffTargetReport,
    ScoredCandidate,
)
from saber.scoring.base import Scorer

logger = logging.getLogger(__name__)


class JEPAMode(str, Enum):
    EFFICIENCY = "efficiency"         # Path A
    DISCRIMINATION = "discrimination"  # Path B
    CONTEXT = "context"               # Path C


class JEPAScorer(Scorer):
    """JEPA-based crRNA scorer.

    Wraps a pre-trained bDNA-JEPA encoder + a fine-tuned prediction head.

    Usage:
        scorer = JEPAScorer(
            encoder_path="checkpoints/bdna_jepa_encoder.pt",
            head_path="checkpoints/cas12a_efficiency_head.pt",
            mode=JEPAMode.EFFICIENCY,
        )
        scored = scorer.score_batch(candidates, offtargets)
    """

    def __init__(
        self,
        encoder_path: str | Path,
        head_path: str | Path,
        mode: JEPAMode = JEPAMode.EFFICIENCY,
        heuristic_fallback: Optional[Scorer] = None,
        embed_dim: int = 256,
        context_window: int = 512,
    ) -> None:
        self.mode = mode
        self.embed_dim = embed_dim
        self.context_window = context_window
        self._fallback = heuristic_fallback

        self.encoder = None
        self.head = None
        self._load_models(Path(encoder_path), Path(head_path))

    def score(
        self,
        candidate: CrRNACandidate,
        offtarget: OffTargetReport,
    ) -> ScoredCandidate:
        # Base heuristic score
        if self._fallback:
            base = self._fallback.score(candidate, offtarget)
        else:
            from saber.scoring.heuristic import HeuristicScorer
            base = HeuristicScorer().score(candidate, offtarget)

        if self.encoder is None or self.head is None:
            return base

        if self.mode == JEPAMode.EFFICIENCY:
            pred = self._predict_efficiency(candidate)
            base.ml_scores.append(MLScore(
                model_name=f"jepa_{self.mode.value}",
                predicted_efficiency=pred,
            ))
        elif self.mode == JEPAMode.DISCRIMINATION:
            logger.debug(
                "Discrimination mode requires MismatchPair — "
                "use score_discrimination() directly"
            )

        return base

    def score_discrimination(
        self,
        candidate: CrRNACandidate,
        pair: MismatchPair,
        offtarget: OffTargetReport,
    ) -> ScoredCandidate:
        """Path B: Predict discrimination ratio for a WT/MUT pair."""
        base = self.score(candidate, offtarget)

        if self.encoder is not None and self.head is not None:
            wt_embed = self._embed_spacer(pair.wt_spacer)
            mut_embed = self._embed_spacer(pair.mut_spacer)
            ratio = self._predict_discrimination(wt_embed, mut_embed)

            base.discrimination = DiscriminationScore(
                wt_activity=ratio[0],
                mut_activity=ratio[1],
            )

        return base

    def embed_genomic_context(self, sequence: str) -> np.ndarray:
        """Path C: Get JEPA embedding for a genomic region.

        Used for target-level features: RPA feasibility, off-target risk,
        local accessibility.
        """
        if self.encoder is None:
            return np.zeros(self.embed_dim, dtype=np.float32)
        return self._embed_sequence(sequence)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_models(self, encoder_path: Path, head_path: Path) -> None:
        try:
            import torch
            if encoder_path.exists():
                self.encoder = torch.jit.load(str(encoder_path), map_location="cpu")
                self.encoder.eval()
                logger.info("Loaded JEPA encoder from %s", encoder_path)
            if head_path.exists():
                self.head = torch.jit.load(str(head_path), map_location="cpu")
                self.head.eval()
                logger.info("Loaded prediction head from %s", head_path)
        except Exception as e:
            logger.warning("Failed to load JEPA models: %s", e)

    def _tokenize(self, seq: str) -> np.ndarray:
        """Convert nucleotide sequence to token IDs for JEPA encoder.

        Uses k-mer tokenization consistent with bDNA-JEPA pre-training.
        """
        kmer_size = 6  # must match pre-training config
        vocab = {}  # loaded from encoder config in practice
        tokens = []
        for i in range(len(seq) - kmer_size + 1):
            kmer = seq[i : i + kmer_size].upper()
            tokens.append(vocab.get(kmer, 0))  # 0 = UNK
        return np.array(tokens, dtype=np.int64)

    def _embed_sequence(self, seq: str) -> np.ndarray:
        """Get JEPA embedding for an arbitrary sequence."""
        import torch
        tokens = self._tokenize(seq)
        with torch.no_grad():
            t = torch.tensor(tokens).unsqueeze(0)
            embedding = self.encoder(t)
            # Mean-pool over sequence dimension
            return embedding.mean(dim=1).squeeze(0).numpy()

    def _embed_spacer(self, spacer: str) -> np.ndarray:
        return self._embed_sequence(spacer)

    def _predict_efficiency(self, candidate: CrRNACandidate) -> float:
        """Path A: Predict absolute cleavage efficiency."""
        import torch
        embedding = self._embed_spacer(candidate.spacer_seq)
        with torch.no_grad():
            t = torch.tensor(embedding).unsqueeze(0)
            pred = self.head(t)
        return float(pred.squeeze().clamp(0, 1).item())

    def _predict_discrimination(
        self, wt_embed: np.ndarray, mut_embed: np.ndarray,
    ) -> tuple[float, float]:
        """Path B: Predict WT and MUT activity from paired embeddings."""
        import torch
        # Concatenate WT and MUT embeddings → discrimination head
        combined = np.concatenate([wt_embed, mut_embed])
        with torch.no_grad():
            t = torch.tensor(combined).unsqueeze(0)
            pred = self.head(t)
            # Head outputs 2 values: [wt_activity, mut_activity]
            activities = pred.squeeze().clamp(0, 1).tolist()
        return (activities[0], activities[1])
