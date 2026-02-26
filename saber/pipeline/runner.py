"""Main pipeline orchestration.

Wires modules 1-7 together. A single call to `run()` executes the
full design workflow: targets → candidates → screening → scoring → panel.

Key integration points:
  - Scanner: initialised with Cas12a variant only; spacer lengths
    come from the scanner's built-in config (multi-length by default).
    NEVER override scanner lengths from pipeline config — this caused
    the PAM desert failure in high-GC genomes.
  - Filter: initialised with OrganismPreset + Cas12aVariant. Thresholds
    (GC, MFE, homopolymer) come from the organism preset, NOT from
    pipeline config. This ensures M.tb gets GC=[0.40, 0.85] instead of
    the default [0.30, 0.70].
  - Mutation type bridge: the resolver classifies mutations into
    detailed types (AA_SUBSTITUTION, RRNA, PROMOTER, etc.), which are
    mapped to the filter's simpler MutationType for adaptive constraints
    (e.g. deletions skip seed requirement).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from saber.core.config import PipelineConfig
from saber.core.types import (
    CrRNACandidate,
    DetectionStrategy,
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
        results = pipeline.run(mutations)
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-init modules
        self._resolver = None
        self._scanner = None
        self._filter = None
        self._mismatch_gen = None
        self._screener = None
        self._scorer = None
        self._optimizer = None
        self._primer_designer = None
        self._mutation_classifier = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, mutations: list[Mutation]) -> dict[str, list[ScoredCandidate]]:
        """Execute the full pipeline: mutations → scored, ranked candidates."""
        logger.info("=== SABER Pipeline: %s ===", self.config.name)

        # Module 1: Resolve targets
        targets = self.resolver.resolve_all(mutations, validate=True)
        self._save("targets.json", [t.model_dump() for t in targets])

        if not targets:
            logger.warning("No targets resolved — stopping")
            self._print_summary({})
            return {}

        # Module 2: Scan + filter candidates
        all_candidates: dict[str, list[CrRNACandidate]] = {}
        all_pairs: dict[str, list[MismatchPair]] = {}

        for target in targets:
            # Scan (no filtering here — scanner maximises pool)
            # Use scan_detailed to separate direct/proximity candidates
            scan_result = self.scanner.scan_detailed(target)
            raw = scan_result.all_candidates

            if scan_result.pam_desert:
                logger.warning(
                    "PAM DESERT for %s: 0 direct candidates, %d proximity candidates",
                    target.label,
                    len(scan_result.proximity_candidates),
                )

            # Classify mutation type for adaptive filtering
            filter_mut_type = self._bridge_mutation_type(target.mutation)

            # Filter with organism-aware + mutation-type-aware constraints
            filtered = self.filter.filter_batch(raw, mutation_type=filter_mut_type)

            all_candidates[target.label] = filtered

            # Generate mismatch pairs for discrimination analysis
            pairs = []
            for c in filtered:
                try:
                    pairs.append(self.mismatch_gen.generate(c, target))
                except Exception as e:
                    logger.debug("Mismatch generation failed for %s: %s", c.candidate_id, e)
            all_pairs[target.label] = pairs

            # Count direct vs proximity in filtered results
            n_direct = sum(1 for c in filtered if c.is_direct)
            n_prox = sum(1 for c in filtered if c.is_proximity)

            logger.info(
                "Target %s: %d raw → %d filtered (%d direct, %d proximity), %d mismatch pairs",
                target.label, len(raw), len(filtered), n_direct, n_prox, len(pairs),
            )

            # Diagnostic logging when everything is rejected
            if not filtered and raw:
                report = self.filter.last_report
                if report:
                    logger.warning(
                        "All %d candidates rejected for %s:\n%s",
                        len(raw), target.label, report.summary(),
                    )

        self._save("candidates.json", {
            label: [c.model_dump() for c in cands]
            for label, cands in all_candidates.items()
        })

        # Module 3: Off-target screening
        all_ot: dict[str, list[OffTargetReport]] = {}
        for label, cands in all_candidates.items():
            if not cands:
                all_ot[label] = []
                continue
            reports = self.screener.screen_batch(cands)
            all_ot[label] = reports
            clean = sum(1 for r in reports if r.is_clean)
            logger.info("Target %s: %d/%d clean (off-target)", label, clean, len(reports))

        # Module 4: Score and rank
        results: dict[str, list[ScoredCandidate]] = {}
        for label in all_candidates:
            if not all_candidates[label]:
                results[label] = []
                continue
            scored = self.scorer.score_batch(
                all_candidates[label], all_ot[label],
            )
            results[label] = scored
            if scored:
                top = scored[0]
                logger.info(
                    "Target %s: top=%s score=%.3f GC=%.0f%% seed_pos=%s strategy=%s",
                    label,
                    top.candidate.candidate_id,
                    top.heuristic.composite,
                    top.candidate.gc_content * 100,
                    top.candidate.mutation_position_in_spacer,
                    top.candidate.detection_strategy.value,
                )

        self._save("scored_candidates.json", {
            label: [s.model_dump() for s in scored]
            for label, scored in results.items()
        })

        total = sum(len(v) for v in results.values())
        logger.info(
            "=== Pipeline complete: %d scored candidates across %d targets ===",
            total, len(targets),
        )
        self._print_summary(results)

        return results

    def run_multiplex(
        self,
        targets: list[Target],
        scored: dict[str, list[ScoredCandidate]],
    ) -> list[ScoredCandidate]:
        """Module 5+6: Optimise multiplex panel from pre-scored candidates."""
        selected = self.optimizer.optimize(targets, scored)
        self._save("panel_selection.json", [
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
        """Scanner with Cas12a variant only — lengths from scanner config.

        IMPORTANT: do NOT pass spacer_length from pipeline config here.
        The scanner's built-in multi-length defaults (18-23 nt) are
        essential to avoid zero candidates in PAM-poor regions.
        """
        if self._scanner is None:
            from saber.candidates.scanner import PAMScanner

            # Determine variant from config
            use_en = getattr(self.config.candidates, "use_enascas12a", True)
            cas_name = getattr(self.config.candidates, "cas_variant", None)

            if cas_name:
                variant = cas_name
            elif use_en:
                variant = "enAsCas12a"
            else:
                variant = "AsCas12a"

            self._scanner = PAMScanner(cas_variant=variant)
        return self._scanner

    @property
    def filter(self):
        """Filter with organism preset — thresholds from preset, NOT config.

        IMPORTANT: do NOT pass gc_min, gc_max, mfe_threshold from
        pipeline config. The organism preset (e.g. M.tb: GC=[0.40, 0.85],
        MFE=-5.0) is calibrated for the target organism and overrides
        any generic defaults in the config file.
        """
        if self._filter is None:
            from saber.candidates.filters import (
                CandidateFilter,
                Cas12aVariant,
                OrganismPreset,
            )

            # Determine organism
            org_name = getattr(self.config, "organism", "mtb")
            preset_map = {p.value: p for p in OrganismPreset}
            organism = preset_map.get(
                org_name, OrganismPreset.MYCOBACTERIUM_TUBERCULOSIS,
            )

            # Determine Cas12a variant
            cas_name = getattr(self.config.candidates, "cas_variant", None)
            if cas_name:
                variant_map = {v.value: v for v in Cas12aVariant}
                cas_variant = variant_map.get(cas_name, Cas12aVariant.enAsCas12a)
            else:
                use_en = getattr(self.config.candidates, "use_enascas12a", True)
                cas_variant = (
                    Cas12aVariant.enAsCas12a if use_en
                    else Cas12aVariant.AsCas12a
                )

            self._filter = CandidateFilter(
                organism=organism,
                cas_variant=cas_variant,
            )
        return self._filter

    @property
    def mutation_classifier(self):
        if self._mutation_classifier is None:
            from saber.targets.resolver import MutationClassifier
            self._mutation_classifier = MutationClassifier()
        return self._mutation_classifier

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
                mtb_index=(
                    self.config.reference.genome_index
                    or self.config.reference.genome_fasta
                ),
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

    def _bridge_mutation_type(self, mutation: Mutation):
        """Map resolver MutationType → filter MutationType.

        The resolver classifies into 10 detailed types; the filter
        uses 8 simpler types. This bridge translates between them.
        Returns None if classification fails (filter uses defaults).
        """
        try:
            from saber.candidates.filters import MutationType as FMT
            from saber.targets.resolver import MutationType as RMT

            classified = self.mutation_classifier.classify(mutation)

            mapping = {
                RMT.AA_SUBSTITUTION: FMT.SNP,
                RMT.NUCLEOTIDE_SNP: FMT.SNP,
                RMT.INSERTION: FMT.INSERTION,
                RMT.DELETION: FMT.DELETION,
                RMT.LARGE_DELETION: FMT.LARGE_DELETION,
                RMT.MNV: FMT.MNV,
                RMT.PROMOTER: FMT.PROMOTER,
                RMT.RRNA: FMT.RRNA,
                RMT.FRAMESHIFT: FMT.FRAMESHIFT,
                RMT.INTERGENIC: FMT.SNP,
                RMT.UNKNOWN: FMT.SNP,
            }
            return mapping.get(classified.mutation_type, FMT.SNP)

        except Exception as e:
            logger.debug("Mutation type classification failed: %s", e)
            return None

    def _print_summary(self, results: dict[str, list[ScoredCandidate]]) -> None:
        """Human-readable results table to stdout."""
        print("=" * 60)
        if not results:
            print("No results.")
            print("=" * 60)
            return

        for label, scored in results.items():
            n_direct = sum(1 for s in scored if s.candidate.is_direct)
            n_prox = sum(1 for s in scored if s.candidate.is_proximity)
            print(f"Target: {label} — {len(scored)} candidates ({n_direct} direct, {n_prox} proximity)")
            for i, s in enumerate(scored[:5]):
                c = s.candidate
                pos = c.mutation_position_in_spacer
                if c.is_proximity:
                    loc = f"PROX({c.proximity_distance}bp)"
                elif c.in_seed:
                    loc = "SEED"
                else:
                    loc = f"pos{pos}"
                print(
                    f"  #{i+1}  {c.spacer_seq}  "
                    f"score={s.heuristic.composite:.3f}  "
                    f"GC={c.gc_content:.0%}  "
                    f"{loc}  "
                    f"PAM={c.pam_seq}  "
                    f"[{c.detection_strategy.value}]"
                )
            if len(scored) > 5:
                print(f"  ... and {len(scored) - 5} more")
        print("=" * 60)

    def _save(self, filename: str, data: object) -> None:
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.debug("Saved %s", path)
