"""Main pipeline orchestration.

Wires modules 1-7 together. A single call to `run()` executes the
full design workflow: targets → candidates → screening → scoring → panel.

Each intermediate result is serialised to the output directory for
reproducibility and debugging.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from saber.core.config import PipelineConfig
from saber.core.types import (
    CrRNACandidate,
    MismatchPair,
    Mutation,
    OffTargetReport,
    ScoredCandidate,
    Target,
)

logger = logging.getLogger(__name__)


class SABERPipeline:
    """End-to-end crRNA design pipeline.

    Usage:
        config = PipelineConfig.from_yaml("configs/mdr_14plex.yaml")
        pipeline = SABERPipeline(config)
        panel = pipeline.run(mutations)
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-init modules — only instantiate what's needed
        self._resolver = None
        self._scanner = None
        self._filter = None
        self._mismatch_gen = None
        self._screener = None
        self._scorer = None
        self._optimizer = None
        self._primer_designer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, mutations: list[Mutation]) -> dict[str, list[ScoredCandidate]]:
        """Execute the full pipeline: mutations → scored, ranked candidates.

        Returns {target_label: [ScoredCandidate, ranked]}.
        """
        logger.info("=== SABER Pipeline: %s ===", self.config.name)

        # Module 1: Resolve targets
        targets = self._resolve_targets(mutations)
        self._save_json("targets.json", [t.model_dump() for t in targets])

        # Module 2: Generate candidates
        all_candidates: dict[str, list[CrRNACandidate]] = {}
        all_pairs: dict[str, list[MismatchPair]] = {}

        for target in targets:
            raw = self.scanner.scan(target)
            filtered = self.filter.filter_batch(raw)
            all_candidates[target.label] = filtered

            # Generate mismatch pairs
            pairs = [self.mismatch_gen.generate(c, target) for c in filtered]
            all_pairs[target.label] = pairs

            logger.info(
                "Target %s: %d raw → %d filtered candidates, %d mismatch pairs",
                target.label, len(raw), len(filtered), len(pairs),
            )

        self._save_json("candidates.json", {
            label: [c.model_dump() for c in cands]
            for label, cands in all_candidates.items()
        })

        # Module 3: Off-target screening
        all_offtargets: dict[str, list[OffTargetReport]] = {}
        for label, cands in all_candidates.items():
            reports = self.screener.screen_batch(cands)
            all_offtargets[label] = reports

            clean = sum(1 for r in reports if r.is_clean)
            logger.info("Target %s: %d/%d clean (off-target)", label, clean, len(reports))

        # Module 4: Score and rank
        results: dict[str, list[ScoredCandidate]] = {}
        for label in all_candidates:
            scored = self.scorer.score_batch(
                all_candidates[label],
                all_offtargets[label],
            )
            results[label] = scored

            if scored:
                top = scored[0]
                logger.info(
                    "Target %s: top candidate %s (score=%.3f)",
                    label, top.candidate.candidate_id, top.heuristic.composite,
                )

        self._save_json("scored_candidates.json", {
            label: [s.model_dump() for s in scored]
            for label, scored in results.items()
        })

        total = sum(len(v) for v in results.values())
        logger.info("=== Pipeline complete: %d scored candidates across %d targets ===", total, len(targets))

        return results

    def run_multiplex(
        self,
        targets: list[Target],
        scored: dict[str, list[ScoredCandidate]],
    ) -> list[ScoredCandidate]:
        """Module 5: Optimise a multiplex panel from pre-scored candidates."""
        selected = self.optimizer.optimize(targets, scored)

        self._save_json("panel_selection.json", [
            {
                "target": s.candidate.target_label,
                "candidate_id": s.candidate.candidate_id,
                "spacer": s.candidate.spacer_seq,
                "rank": s.rank,
                "score": s.heuristic.composite,
            }
            for s in selected
        ])

        return selected

    # ------------------------------------------------------------------
    # Module accessors (lazy init)
    # ------------------------------------------------------------------

    @property
    def resolver(self):
        if self._resolver is None:
            from saber.targets.resolver import TargetResolver
            self._resolver = TargetResolver(
                fasta=self.config.reference.genome_fasta,
                gff=self.config.reference.gff_annotation,
            )
        return self._resolver

    @property
    def scanner(self):
        if self._scanner is None:
            from saber.candidates.scanner import PAMScanner
            self._scanner = PAMScanner(
                spacer_length=self.config.candidates.spacer_lengths[0],
                use_enascas12a=self.config.candidates.use_enascas12a,
            )
        return self._scanner

    @property
    def filter(self):
        if self._filter is None:
            from saber.candidates.filters import CandidateFilter
            cfg = self.config.candidates
            self._filter = CandidateFilter(
                gc_min=cfg.gc_min,
                gc_max=cfg.gc_max,
                homopolymer_max=cfg.homopolymer_max,
                mfe_threshold=cfg.mfe_threshold,
                require_seed=cfg.require_seed_mutation,
            )
        return self._filter

    @property
    def mismatch_gen(self):
        if self._mismatch_gen is None:
            from saber.candidates.mismatch import MismatchGenerator
            self._mismatch_gen = MismatchGenerator()
        return self._mismatch_gen

    @property
    def screener(self):
        if self._screener is None:
            from saber.offtarget.screener import OffTargetScreener
            self._screener = OffTargetScreener(
                mtb_index=self.config.reference.genome_index or self.config.reference.genome_fasta,
                human_index=self.config.reference.human_index,
            )
        return self._screener

    @property
    def scorer(self):
        if self._scorer is None:
            cfg = self.config.scoring
            if cfg.use_ml and cfg.ml_model_path:
                from saber.scoring.sequence_ml import SequenceMLScorer
                from saber.scoring.heuristic import HeuristicScorer
                self._scorer = SequenceMLScorer(
                    model_path=cfg.ml_model_path,
                    heuristic_fallback=HeuristicScorer(),
                )
            else:
                from saber.scoring.heuristic import HeuristicScorer
                self._scorer = HeuristicScorer()
        return self._scorer

    @property
    def optimizer(self):
        if self._optimizer is None:
            from saber.multiplex.optimizer import MultiplexOptimizer
            cfg = self.config.multiplex
            self._optimizer = MultiplexOptimizer(
                max_iterations=cfg.max_iterations,
            )
        return self._optimizer

    @property
    def primer_designer(self):
        if self._primer_designer is None:
            from saber.primers.rpa_designer import RPAPrimerDesigner
            self._primer_designer = RPAPrimerDesigner()
        return self._primer_designer

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _resolve_targets(self, mutations: list[Mutation]) -> list[Target]:
        return self.resolver.resolve_all(mutations)

    def _save_json(self, filename: str, data: object) -> None:
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.debug("Saved %s", path)
