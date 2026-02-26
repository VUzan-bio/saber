"""Hard filters for crRNA candidates — organism-aware, mutation-type-aware.

Supports arbitrary bacterial genomes (not just M.tb), multiple Cas12a
variants (AsCas12a, enAsCas12a, LbCas12a, FnCas12a), and all clinically
relevant mutation types (SNP, MNV, insertion, deletion, promoter).

Filter cascade is ordered by computational cost (cheapest first):
  1. Spacer length           — O(1)
  2. Mutation in seed region — O(1)
  3. GC content (adaptive)   — O(n)
  4. Homopolymer runs        — O(n)
  5. Poly-T terminator       — O(n)
  6. Low-complexity (dinuc)  — O(n)
  7. Self-complementarity    — O(n²)
  8. Secondary structure MFE — O(n³), external call

Design principles:
  - Every threshold is configurable at init, with organism-aware defaults
  - Filters return structured diagnostics (not just pass/fail)
  - Batch filtering produces a FilterReport with per-filter rejection stats
  - Soft mode: score penalties instead of hard rejection (for ranking)
  - All filters are independently toggleable

References:
  - Kim et al., Nature Biotechnology 2018 (Cas12a design rules)
  - Creutzburg et al., NAR 2020 (enAsCas12a expanded PAM)
  - Wessels et al., Nature Biotechnology 2020 (Cas13 design rules, poly-U)
  - Konstantakos et al., NAR 2022 (CRISPRloci filter benchmarks)
  - Tycko et al., Nature Methods 2023 (CRISPR guide design review)
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from saber.core.types import CrRNACandidate

logger = logging.getLogger(__name__)


# ======================================================================
# Organism presets
# ======================================================================

class OrganismPreset(str, Enum):
    """Pre-configured filter thresholds for common target organisms."""
    MYCOBACTERIUM_TUBERCULOSIS = "mtb"
    ESCHERICHIA_COLI = "ecoli"
    STAPHYLOCOCCUS_AUREUS = "saureus"
    PSEUDOMONAS_AERUGINOSA = "paeruginosa"
    KLEBSIELLA_PNEUMONIAE = "kpneumoniae"
    NEISSERIA_GONORRHOEAE = "ngonorrhoeae"
    ACINETOBACTER_BAUMANNII = "abaumannii"
    SALMONELLA_ENTERICA = "senterica"
    ENTEROCOCCUS_FAECIUM = "efaecium"
    GENERIC_HIGH_GC = "high_gc"       # GC > 60%
    GENERIC_MEDIUM_GC = "medium_gc"   # GC 40-60%
    GENERIC_LOW_GC = "low_gc"         # GC < 40%
    CUSTOM = "custom"


@dataclass(frozen=True)
class OrganismParams:
    """Organism-specific biophysical parameters for filter calibration."""
    name: str
    genome_gc: float                     # genome-wide GC fraction
    gc_spacer_min: float                 # minimum spacer GC (absolute)
    gc_spacer_max: float                 # maximum spacer GC (absolute)
    gc_deviation_max: float              # max deviation from genome GC
    homopolymer_max: int                 # max tolerated homopolymer run
    mfe_threshold: float                 # kcal/mol, below this = too structured
    poly_t_max: int                      # max consecutive T's (Pol III terminator)
    dinucleotide_fraction_max: float     # max fraction of any dinucleotide repeat
    self_comp_max: int                   # max self-complementary stretch (nt)


# Curated defaults per organism
# GC windows calibrated to ±15-20% of genome GC, clamped to [0.20, 0.85]
# MFE thresholds relaxed for high-GC organisms (more stable by default)
ORGANISM_PRESETS: dict[OrganismPreset, OrganismParams] = {
    OrganismPreset.MYCOBACTERIUM_TUBERCULOSIS: OrganismParams(
        name="Mycobacterium tuberculosis",
        genome_gc=0.656,
        gc_spacer_min=0.40,
        gc_spacer_max=0.85,
        gc_deviation_max=0.20,
        homopolymer_max=4,
        mfe_threshold=-5.0,       # relaxed: high-GC means more structure
        poly_t_max=4,
        dinucleotide_fraction_max=0.45,
        self_comp_max=8,
    ),
    OrganismPreset.ESCHERICHIA_COLI: OrganismParams(
        name="Escherichia coli",
        genome_gc=0.508,
        gc_spacer_min=0.35,
        gc_spacer_max=0.70,
        gc_deviation_max=0.20,
        homopolymer_max=4,
        mfe_threshold=-3.0,
        poly_t_max=4,
        dinucleotide_fraction_max=0.40,
        self_comp_max=8,
    ),
    OrganismPreset.STAPHYLOCOCCUS_AUREUS: OrganismParams(
        name="Staphylococcus aureus",
        genome_gc=0.328,
        gc_spacer_min=0.20,
        gc_spacer_max=0.55,
        gc_deviation_max=0.20,
        homopolymer_max=5,
        mfe_threshold=-2.0,       # low GC = less secondary structure
        poly_t_max=4,
        dinucleotide_fraction_max=0.40,
        self_comp_max=8,
    ),
    OrganismPreset.PSEUDOMONAS_AERUGINOSA: OrganismParams(
        name="Pseudomonas aeruginosa",
        genome_gc=0.663,
        gc_spacer_min=0.40,
        gc_spacer_max=0.85,
        gc_deviation_max=0.20,
        homopolymer_max=4,
        mfe_threshold=-5.0,
        poly_t_max=4,
        dinucleotide_fraction_max=0.45,
        self_comp_max=8,
    ),
    OrganismPreset.KLEBSIELLA_PNEUMONIAE: OrganismParams(
        name="Klebsiella pneumoniae",
        genome_gc=0.573,
        gc_spacer_min=0.35,
        gc_spacer_max=0.75,
        gc_deviation_max=0.20,
        homopolymer_max=4,
        mfe_threshold=-4.0,
        poly_t_max=4,
        dinucleotide_fraction_max=0.40,
        self_comp_max=8,
    ),
    OrganismPreset.NEISSERIA_GONORRHOEAE: OrganismParams(
        name="Neisseria gonorrhoeae",
        genome_gc=0.525,
        gc_spacer_min=0.35,
        gc_spacer_max=0.70,
        gc_deviation_max=0.20,
        homopolymer_max=4,
        mfe_threshold=-3.0,
        poly_t_max=4,
        dinucleotide_fraction_max=0.40,
        self_comp_max=8,
    ),
    OrganismPreset.ACINETOBACTER_BAUMANNII: OrganismParams(
        name="Acinetobacter baumannii",
        genome_gc=0.391,
        gc_spacer_min=0.25,
        gc_spacer_max=0.60,
        gc_deviation_max=0.20,
        homopolymer_max=5,
        mfe_threshold=-2.5,
        poly_t_max=4,
        dinucleotide_fraction_max=0.40,
        self_comp_max=8,
    ),
    OrganismPreset.SALMONELLA_ENTERICA: OrganismParams(
        name="Salmonella enterica",
        genome_gc=0.522,
        gc_spacer_min=0.35,
        gc_spacer_max=0.70,
        gc_deviation_max=0.20,
        homopolymer_max=4,
        mfe_threshold=-3.0,
        poly_t_max=4,
        dinucleotide_fraction_max=0.40,
        self_comp_max=8,
    ),
    OrganismPreset.ENTEROCOCCUS_FAECIUM: OrganismParams(
        name="Enterococcus faecium",
        genome_gc=0.379,
        gc_spacer_min=0.25,
        gc_spacer_max=0.60,
        gc_deviation_max=0.20,
        homopolymer_max=5,
        mfe_threshold=-2.5,
        poly_t_max=4,
        dinucleotide_fraction_max=0.40,
        self_comp_max=8,
    ),
    OrganismPreset.GENERIC_HIGH_GC: OrganismParams(
        name="Generic high-GC organism",
        genome_gc=0.65,
        gc_spacer_min=0.40,
        gc_spacer_max=0.85,
        gc_deviation_max=0.25,
        homopolymer_max=4,
        mfe_threshold=-5.0,
        poly_t_max=4,
        dinucleotide_fraction_max=0.45,
        self_comp_max=8,
    ),
    OrganismPreset.GENERIC_MEDIUM_GC: OrganismParams(
        name="Generic medium-GC organism",
        genome_gc=0.50,
        gc_spacer_min=0.30,
        gc_spacer_max=0.70,
        gc_deviation_max=0.20,
        homopolymer_max=4,
        mfe_threshold=-3.0,
        poly_t_max=4,
        dinucleotide_fraction_max=0.40,
        self_comp_max=8,
    ),
    OrganismPreset.GENERIC_LOW_GC: OrganismParams(
        name="Generic low-GC organism",
        genome_gc=0.35,
        gc_spacer_min=0.20,
        gc_spacer_max=0.55,
        gc_deviation_max=0.20,
        homopolymer_max=5,
        mfe_threshold=-2.0,
        poly_t_max=5,
        dinucleotide_fraction_max=0.40,
        self_comp_max=8,
    ),
}


def params_from_genome_gc(gc: float, name: str = "custom") -> OrganismParams:
    """Auto-generate filter parameters from observed genome GC content.

    Uses empirically calibrated scaling rules:
    - GC window: genome_gc ± 0.20, clamped to [0.20, 0.85]
    - MFE scales with GC: high-GC genomes tolerate more structure
    - Homopolymer threshold: 4 for high-GC, 5 for low-GC (A/T runs more common)
    """
    gc_min = max(0.20, gc - 0.20)
    gc_max = min(0.85, gc + 0.20)
    mfe = -2.0 + (-3.0 * gc)  # linear: -2 at GC=0, -5 at GC≈1
    homopolymer = 4 if gc > 0.45 else 5

    return OrganismParams(
        name=name,
        genome_gc=gc,
        gc_spacer_min=gc_min,
        gc_spacer_max=gc_max,
        gc_deviation_max=0.20,
        homopolymer_max=homopolymer,
        mfe_threshold=round(mfe, 1),
        poly_t_max=4 if gc > 0.45 else 5,
        dinucleotide_fraction_max=0.40,
        self_comp_max=8,
    )


# ======================================================================
# Cas12a variant profiles
# ======================================================================

class Cas12aVariant(str, Enum):
    """Supported Cas12a orthologs / engineered variants."""
    AsCas12a = "AsCas12a"
    enAsCas12a = "enAsCas12a"
    LbCas12a = "LbCas12a"
    FnCas12a = "FnCas12a"
    Cas12a_ultra = "Cas12a_ultra"  # AsCas12a Ultra (Kleinstiver lab)


@dataclass(frozen=True)
class Cas12aProfile:
    """Variant-specific guide design parameters."""
    seed_start: int          # seed region start (1-based from PAM-proximal)
    seed_end: int            # seed region end (inclusive)
    spacer_length: int       # canonical spacer length
    spacer_length_range: tuple[int, int]   # min, max tolerated
    gc_optimum: float        # GC fraction with highest measured activity
    mfe_sensitivity: float   # multiplier on MFE threshold (higher = more sensitive)


CAS12A_PROFILES: dict[Cas12aVariant, Cas12aProfile] = {
    Cas12aVariant.AsCas12a: Cas12aProfile(
        seed_start=1, seed_end=8,
        spacer_length=20,
        spacer_length_range=(18, 23),
        gc_optimum=0.50,
        mfe_sensitivity=1.0,
    ),
    Cas12aVariant.enAsCas12a: Cas12aProfile(
        seed_start=1, seed_end=8,
        spacer_length=20,
        spacer_length_range=(18, 23),
        gc_optimum=0.50,
        mfe_sensitivity=1.0,
    ),
    Cas12aVariant.LbCas12a: Cas12aProfile(
        seed_start=1, seed_end=10,       # slightly extended seed
        spacer_length=23,                 # LbCas12a prefers 23 nt
        spacer_length_range=(19, 25),
        gc_optimum=0.45,
        mfe_sensitivity=1.2,             # more sensitive to structure
    ),
    Cas12aVariant.FnCas12a: Cas12aProfile(
        seed_start=1, seed_end=6,        # shorter critical seed
        spacer_length=20,
        spacer_length_range=(18, 24),
        gc_optimum=0.50,
        mfe_sensitivity=0.8,
    ),
    Cas12aVariant.Cas12a_ultra: Cas12aProfile(
        seed_start=1, seed_end=8,
        spacer_length=20,
        spacer_length_range=(18, 23),
        gc_optimum=0.55,                 # Ultra tolerates slightly higher GC
        mfe_sensitivity=0.9,
    ),
}


# ======================================================================
# Mutation type constraints
# ======================================================================

class MutationType(str, Enum):
    """Classification of mutations for filter logic."""
    SNP = "snp"                    # single nucleotide polymorphism
    MNV = "mnv"                    # multi-nucleotide variant (2-3 nt)
    INSERTION = "insertion"
    DELETION = "deletion"
    PROMOTER = "promoter"          # promoter region (e.g. inhA C-15T)
    RRNA = "rrna"                  # rRNA mutations (e.g. rrs A1401G)
    FRAMESHIFT = "frameshift"      # frameshifts causing LOF
    LARGE_DELETION = "large_del"   # large deletions (e.g. katG full deletion)


@dataclass(frozen=True)
class MutationConstraints:
    """Mutation-type-specific filter adjustments.

    Some mutations need relaxed or tightened constraints:
    - SNPs: strict seed requirement (mutation must be in seed for discrimination)
    - Deletions: seed less critical (absence vs presence = strong signal)
    - Promoter: may be far from coding region, different PAM landscape
    - rRNA: high conservation, fewer off-target concerns
    """
    require_seed: bool           # mutation must be in seed region
    seed_extension: int          # extra positions beyond canonical seed (for MNV)
    gc_tolerance_extra: float    # additional GC tolerance for this mutation type
    min_discrimination_score: float  # minimum expected WT/MUT discrimination


MUTATION_CONSTRAINTS: dict[MutationType, MutationConstraints] = {
    MutationType.SNP: MutationConstraints(
        require_seed=True,
        seed_extension=0,
        gc_tolerance_extra=0.0,
        min_discrimination_score=2.0,
    ),
    MutationType.MNV: MutationConstraints(
        require_seed=True,
        seed_extension=2,         # MNV can span seed boundary
        gc_tolerance_extra=0.0,
        min_discrimination_score=3.0,  # MNV should discriminate better
    ),
    MutationType.INSERTION: MutationConstraints(
        require_seed=False,       # length change is strong enough signal
        seed_extension=0,
        gc_tolerance_extra=0.05,
        min_discrimination_score=5.0,
    ),
    MutationType.DELETION: MutationConstraints(
        require_seed=False,       # absence is strong signal
        seed_extension=0,
        gc_tolerance_extra=0.05,
        min_discrimination_score=5.0,
    ),
    MutationType.PROMOTER: MutationConstraints(
        require_seed=True,
        seed_extension=0,
        gc_tolerance_extra=0.05,   # promoter regions may differ from CDS
        min_discrimination_score=1.5,
    ),
    MutationType.RRNA: MutationConstraints(
        require_seed=True,
        seed_extension=0,
        gc_tolerance_extra=0.0,
        min_discrimination_score=2.0,
    ),
    MutationType.FRAMESHIFT: MutationConstraints(
        require_seed=False,        # any disruption is informative
        seed_extension=0,
        gc_tolerance_extra=0.10,
        min_discrimination_score=10.0,
    ),
    MutationType.LARGE_DELETION: MutationConstraints(
        require_seed=False,        # presence/absence assay
        seed_extension=0,
        gc_tolerance_extra=0.15,
        min_discrimination_score=50.0,
    ),
}


def classify_mutation(mutation) -> MutationType:
    """Infer mutation type from Mutation object attributes.

    Heuristics:
    - Nucleotide change with single base → SNP
    - Nucleotide change with 'ins'/'del' → insertion/deletion
    - Gene is 'rrs' or 'rrl' → rRNA
    - Position is negative (e.g. C-15T) → promoter
    - ref_aa and alt_aa differ by >1 codon → MNV or frameshift
    """
    # Check for explicit nucleotide annotation
    nuc = getattr(mutation, "nucleotide_change", None) or ""
    gene = getattr(mutation, "gene", "")
    position = getattr(mutation, "position", 0)

    # rRNA genes
    if gene.lower() in ("rrs", "rrl", "rrn"):
        return MutationType.RRNA

    # Promoter mutations (negative position or annotated)
    if position < 0 or nuc.startswith("c.") and "-" in nuc[:10]:
        return MutationType.PROMOTER

    # Insertion / deletion from nucleotide annotation
    nuc_lower = nuc.lower()
    if "ins" in nuc_lower:
        return MutationType.INSERTION
    if "del" in nuc_lower:
        # Large deletion: >50 bp or full gene
        if "full" in nuc_lower or _parse_del_size(nuc) > 50:
            return MutationType.LARGE_DELETION
        return MutationType.DELETION

    # Frameshift: alt amino acid is '*' (stop) or annotation contains 'fs'
    alt_aa = getattr(mutation, "alt_aa", "")
    if alt_aa == "*" or "fs" in nuc_lower:
        return MutationType.FRAMESHIFT

    # MNV: multiple nucleotide changes annotated
    if nuc and _count_changes(nuc) > 1:
        return MutationType.MNV

    # Default: SNP
    return MutationType.SNP


def _parse_del_size(nuc: str) -> int:
    """Extract deletion size from nucleotide change string."""
    # Patterns like 'c.123_456del' → 456 - 123 + 1
    match = re.search(r"(\d+)_(\d+)del", nuc)
    if match:
        return int(match.group(2)) - int(match.group(1)) + 1
    return 1


def _count_changes(nuc: str) -> int:
    """Count number of nucleotide substitutions in change string."""
    # Simple heuristic: count '>' characters (e.g. 'c.123A>G;c.124T>C')
    return max(1, nuc.count(">"))


# ======================================================================
# Filter result types
# ======================================================================

class FilterName(str, Enum):
    """Names of individual filters for tracking."""
    SPACER_LENGTH = "spacer_length"
    SEED_REGION = "seed_region"
    GC_CONTENT = "gc_content"
    HOMOPOLYMER = "homopolymer"
    POLY_T = "poly_t_terminator"
    LOW_COMPLEXITY = "low_complexity"
    SELF_COMPLEMENTARITY = "self_complementarity"
    SECONDARY_STRUCTURE = "secondary_structure"


@dataclass
class FilterDecision:
    """Detailed result from a single filter."""
    filter_name: FilterName
    passed: bool
    value: float                        # measured value
    threshold: float                    # threshold applied
    reason: Optional[str] = None        # human-readable explanation
    penalty: float = 0.0               # soft-mode penalty (0 = no penalty)


@dataclass
class CandidateFilterResult:
    """Complete filter evaluation for one candidate."""
    candidate_id: str
    passed: bool
    decisions: list[FilterDecision] = field(default_factory=list)

    @property
    def rejection_reasons(self) -> list[str]:
        return [d.reason for d in self.decisions if not d.passed and d.reason]

    @property
    def total_penalty(self) -> float:
        return sum(d.penalty for d in self.decisions)


@dataclass
class FilterReport:
    """Aggregate statistics from filtering a batch of candidates."""
    total_candidates: int
    passed_count: int
    rejected_count: int
    per_filter_rejections: dict[FilterName, int] = field(default_factory=dict)
    # Candidates that failed ONLY because of a single filter
    # (useful for understanding which filter is the bottleneck)
    single_filter_rejections: dict[FilterName, int] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        return self.passed_count / self.total_candidates if self.total_candidates else 0.0

    def summary(self) -> str:
        lines = [
            f"Filter report: {self.passed_count}/{self.total_candidates} passed "
            f"({self.pass_rate:.1%})",
        ]
        if self.per_filter_rejections:
            lines.append("  Rejections by filter:")
            for fn, count in sorted(
                self.per_filter_rejections.items(), key=lambda x: -x[1]
            ):
                sole = self.single_filter_rejections.get(fn, 0)
                lines.append(f"    {fn.value:30s}: {count:4d} total, {sole:4d} sole cause")
        return "\n".join(lines)


# ======================================================================
# Main filter class
# ======================================================================

class CandidateFilter:
    """Apply biophysical hard filters to crRNA candidates.

    Organism-aware, Cas12a-variant-aware, mutation-type-aware.

    Usage:
        # Preset for M. tuberculosis with AsCas12a:
        filt = CandidateFilter(
            organism=OrganismPreset.MYCOBACTERIUM_TUBERCULOSIS,
            cas_variant=Cas12aVariant.AsCas12a,
        )
        passed = filt.filter_batch(candidates)
        print(filt.last_report.summary())

        # Auto-detect from genome GC:
        filt = CandidateFilter.from_genome_gc(
            gc=0.656, cas_variant=Cas12aVariant.enAsCas12a,
        )

        # Soft mode (score penalties instead of hard reject):
        filt = CandidateFilter(organism=..., soft_mode=True)
        all_candidates = filt.filter_batch(candidates)  # returns all
        # candidates now have filt.last_results[id].total_penalty
    """

    def __init__(
        self,
        organism: OrganismPreset = OrganismPreset.GENERIC_MEDIUM_GC,
        cas_variant: Cas12aVariant = Cas12aVariant.AsCas12a,
        organism_params: Optional[OrganismParams] = None,
        cas_profile: Optional[Cas12aProfile] = None,
        # Override individual thresholds (None = use preset)
        gc_min: Optional[float] = None,
        gc_max: Optional[float] = None,
        homopolymer_max: Optional[int] = None,
        mfe_threshold: Optional[float] = None,
        poly_t_max: Optional[int] = None,
        dinuc_max: Optional[float] = None,
        self_comp_max: Optional[int] = None,
        # Feature toggles
        require_seed: bool = True,
        check_structure: bool = True,
        check_poly_t: bool = True,
        check_low_complexity: bool = True,
        check_self_complementarity: bool = True,
        # Behaviour
        soft_mode: bool = False,
    ) -> None:
        # Resolve organism parameters
        if organism_params is not None:
            self.org = organism_params
        elif organism == OrganismPreset.CUSTOM:
            raise ValueError(
                "OrganismPreset.CUSTOM requires explicit organism_params"
            )
        else:
            self.org = ORGANISM_PRESETS[organism]

        # Resolve Cas12a profile
        self.cas = cas_profile or CAS12A_PROFILES[cas_variant]

        # Apply overrides
        self.gc_min = gc_min if gc_min is not None else self.org.gc_spacer_min
        self.gc_max = gc_max if gc_max is not None else self.org.gc_spacer_max
        self.homopolymer_max = (
            homopolymer_max if homopolymer_max is not None
            else self.org.homopolymer_max
        )
        self.mfe_threshold = (
            mfe_threshold if mfe_threshold is not None
            else self.org.mfe_threshold * self.cas.mfe_sensitivity
        )
        self.poly_t_max = (
            poly_t_max if poly_t_max is not None
            else self.org.poly_t_max
        )
        self.dinuc_max = (
            dinuc_max if dinuc_max is not None
            else self.org.dinucleotide_fraction_max
        )
        self.self_comp_max = (
            self_comp_max if self_comp_max is not None
            else self.org.self_comp_max
        )

        # Feature toggles
        self.require_seed = require_seed
        self.check_structure = check_structure
        self.check_poly_t = check_poly_t
        self.check_low_complexity = check_low_complexity
        self.check_self_complementarity = check_self_complementarity
        self.soft_mode = soft_mode

        # Results storage
        self.last_report: Optional[FilterReport] = None
        self.last_results: dict[str, CandidateFilterResult] = {}

        logger.info(
            "CandidateFilter initialized: organism=%s (GC=%.1f%%), "
            "cas=%s (seed %d-%d), GC range=[%.2f, %.2f], "
            "MFE threshold=%.1f kcal/mol, soft_mode=%s",
            self.org.name, self.org.genome_gc * 100,
            cas_variant if cas_profile is None else "custom",
            self.cas.seed_start, self.cas.seed_end,
            self.gc_min, self.gc_max,
            self.mfe_threshold,
            self.soft_mode,
        )

    @classmethod
    def from_genome_gc(
        cls,
        gc: float,
        cas_variant: Cas12aVariant = Cas12aVariant.AsCas12a,
        name: str = "auto",
        **kwargs,
    ) -> CandidateFilter:
        """Construct filter with auto-calibrated thresholds from genome GC."""
        params = params_from_genome_gc(gc, name)
        return cls(
            organism=OrganismPreset.CUSTOM,
            cas_variant=cas_variant,
            organism_params=params,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(
        self,
        candidate: CrRNACandidate,
        mutation_type: Optional[MutationType] = None,
    ) -> CandidateFilterResult:
        """Evaluate a single candidate against all filters.

        If mutation_type is provided, adjusts seed and GC constraints
        according to MUTATION_CONSTRAINTS.
        """
        decisions: list[FilterDecision] = []
        mc = MUTATION_CONSTRAINTS.get(mutation_type, None) if mutation_type else None

        # Determine effective seed requirement
        effective_require_seed = self.require_seed
        effective_seed_end = self.cas.seed_end
        if mc is not None:
            effective_require_seed = mc.require_seed
            effective_seed_end = self.cas.seed_end + mc.seed_extension

        # Determine effective GC bounds
        gc_extra = mc.gc_tolerance_extra if mc else 0.0
        effective_gc_min = max(0.0, self.gc_min - gc_extra)
        effective_gc_max = min(1.0, self.gc_max + gc_extra)

        spacer = candidate.spacer_seq
        spacer_len = len(spacer)

        # --- Filter 1: Spacer length ---
        min_len, max_len = self.cas.spacer_length_range
        decisions.append(self._check_spacer_length(spacer_len, min_len, max_len))

        # --- Filter 2: Seed region ---
        # SKIP for proximity candidates: mutation is outside the spacer
        # by design. Discrimination comes from AS-RPA primers, not from
        # crRNA mismatch position. Applying seed filter to proximity
        # candidates would reject 100% of them (mutation_position=None).
        is_proximity = getattr(candidate, "is_proximity", False)
        if effective_require_seed and not is_proximity:
            decisions.append(self._check_seed(
                candidate, self.cas.seed_start, effective_seed_end
            ))

        # --- Filter 3: GC content ---
        decisions.append(self._check_gc(candidate.gc_content, effective_gc_min, effective_gc_max))

        # --- Filter 4: Homopolymer ---
        decisions.append(self._check_homopolymer(spacer))

        # --- Filter 5: Poly-T terminator ---
        if self.check_poly_t:
            decisions.append(self._check_poly_t(spacer))

        # --- Filter 6: Low complexity ---
        if self.check_low_complexity:
            decisions.append(self._check_low_complexity(spacer))

        # --- Filter 7: Self-complementarity ---
        if self.check_self_complementarity:
            decisions.append(self._check_self_complementarity(spacer))

        # --- Filter 8: Secondary structure (MFE) ---
        if self.check_structure:
            decisions.append(self._check_mfe(spacer, candidate))

        # Overall pass/fail
        all_passed = all(d.passed for d in decisions)

        return CandidateFilterResult(
            candidate_id=candidate.candidate_id,
            passed=all_passed,
            decisions=decisions,
        )

    def filter_batch(
        self,
        candidates: list[CrRNACandidate],
        mutation_type: Optional[MutationType] = None,
    ) -> list[CrRNACandidate]:
        """Filter a list of candidates, returning those that pass.

        In soft_mode, returns ALL candidates (none rejected), but each
        has penalty scores stored in self.last_results.
        """
        self.last_results = {}
        passed: list[CrRNACandidate] = []
        per_filter: dict[FilterName, int] = {fn: 0 for fn in FilterName}
        single_filter: dict[FilterName, int] = {fn: 0 for fn in FilterName}

        for c in candidates:
            result = self.apply(c, mutation_type)
            self.last_results[c.candidate_id] = result

            if result.passed or self.soft_mode:
                passed.append(c)

            if not result.passed:
                failed_filters = [
                    d.filter_name for d in result.decisions if not d.passed
                ]
                for fn in failed_filters:
                    per_filter[fn] += 1
                if len(failed_filters) == 1:
                    single_filter[failed_filters[0]] += 1

        # Build report
        rejected = len(candidates) - len(passed) if not self.soft_mode else 0
        self.last_report = FilterReport(
            total_candidates=len(candidates),
            passed_count=len(passed),
            rejected_count=rejected,
            per_filter_rejections={k: v for k, v in per_filter.items() if v > 0},
            single_filter_rejections={k: v for k, v in single_filter.items() if v > 0},
        )

        logger.info(self.last_report.summary())
        return passed

    # ------------------------------------------------------------------
    # Individual filter implementations
    # ------------------------------------------------------------------

    def _check_spacer_length(
        self, length: int, min_len: int, max_len: int
    ) -> FilterDecision:
        ok = min_len <= length <= max_len
        return FilterDecision(
            filter_name=FilterName.SPACER_LENGTH,
            passed=ok,
            value=float(length),
            threshold=float(self.cas.spacer_length),
            reason=None if ok else (
                f"spacer length={length}, required [{min_len}-{max_len}]"
            ),
            penalty=0.0 if ok else 1.0,
        )

    def _check_seed(
        self, candidate: CrRNACandidate, seed_start: int, seed_end: int,
    ) -> FilterDecision:
        """Mutation must fall within the seed region for SNV discrimination."""
        pos = candidate.mutation_position_in_spacer
        in_seed = seed_start <= pos <= seed_end if pos is not None else False
        return FilterDecision(
            filter_name=FilterName.SEED_REGION,
            passed=in_seed,
            value=float(pos) if pos is not None else -1.0,
            threshold=float(seed_end),
            reason=None if in_seed else (
                f"mutation at spacer position {pos}, "
                f"outside seed [{seed_start}-{seed_end}]"
            ),
            penalty=0.0 if in_seed else 0.8,
        )

    def _check_gc(
        self, gc: float, gc_min: float, gc_max: float,
    ) -> FilterDecision:
        ok = gc_min <= gc <= gc_max
        # Penalty proportional to distance from acceptable range
        if ok:
            penalty = 0.0
        else:
            overshoot = max(gc - gc_max, gc_min - gc, 0)
            penalty = min(1.0, overshoot / 0.10)  # full penalty at 10% over
        return FilterDecision(
            filter_name=FilterName.GC_CONTENT,
            passed=ok,
            value=gc,
            threshold=gc_max if gc > gc_max else gc_min,
            reason=None if ok else (
                f"GC={gc:.2%}, required [{gc_min:.0%}-{gc_max:.0%}] "
                f"(genome GC={self.org.genome_gc:.1%})"
            ),
            penalty=penalty,
        )

    def _check_homopolymer(self, spacer: str) -> FilterDecision:
        max_run = self._max_homopolymer(spacer)
        ok = max_run <= self.homopolymer_max
        return FilterDecision(
            filter_name=FilterName.HOMOPOLYMER,
            passed=ok,
            value=float(max_run),
            threshold=float(self.homopolymer_max),
            reason=None if ok else (
                f"homopolymer run={max_run}, max={self.homopolymer_max}"
            ),
            penalty=0.0 if ok else 0.3 * (max_run - self.homopolymer_max),
        )

    def _check_poly_t(self, spacer: str) -> FilterDecision:
        """Consecutive T's act as Pol III terminator signal (RNA expression)."""
        max_t = self._max_run(spacer, "T")
        ok = max_t <= self.poly_t_max
        return FilterDecision(
            filter_name=FilterName.POLY_T,
            passed=ok,
            value=float(max_t),
            threshold=float(self.poly_t_max),
            reason=None if ok else (
                f"poly-T run={max_t}, max={self.poly_t_max} "
                f"(Pol III terminator risk)"
            ),
            penalty=0.0 if ok else 0.5,
        )

    def _check_low_complexity(self, spacer: str) -> FilterDecision:
        """Flag spacers dominated by dinucleotide repeats (e.g. GCGCGCGC).

        High dinucleotide repeat fraction → poor discrimination,
        non-specific binding, problematic for RPA amplification.
        """
        frac = self._dinucleotide_repeat_fraction(spacer)
        ok = frac <= self.dinuc_max
        return FilterDecision(
            filter_name=FilterName.LOW_COMPLEXITY,
            passed=ok,
            value=frac,
            threshold=self.dinuc_max,
            reason=None if ok else (
                f"dinucleotide repeat fraction={frac:.2%}, "
                f"max={self.dinuc_max:.0%}"
            ),
            penalty=0.0 if ok else 0.4,
        )

    def _check_self_complementarity(self, spacer: str) -> FilterDecision:
        """Detect self-complementary stretches that could cause dimerization.

        Long self-complementary regions compete with target binding
        and reduce effective concentration.
        """
        max_comp = self._max_self_complement(spacer)
        ok = max_comp <= self.self_comp_max
        return FilterDecision(
            filter_name=FilterName.SELF_COMPLEMENTARITY,
            passed=ok,
            value=float(max_comp),
            threshold=float(self.self_comp_max),
            reason=None if ok else (
                f"self-complementary stretch={max_comp} nt, "
                f"max={self.self_comp_max}"
            ),
            penalty=0.0 if ok else 0.4,
        )

    def _check_mfe(
        self, spacer: str, candidate: CrRNACandidate,
    ) -> FilterDecision:
        """Secondary structure check via ViennaRNA RNAfold."""
        mfe = self._compute_mfe(spacer)
        if mfe is None:
            # ViennaRNA not available — skip gracefully
            return FilterDecision(
                filter_name=FilterName.SECONDARY_STRUCTURE,
                passed=True,
                value=0.0,
                threshold=self.mfe_threshold,
                reason=None,
            )

        # Store on candidate for downstream scoring
        candidate.mfe = mfe
        ok = mfe >= self.mfe_threshold  # less negative = less structure = better
        penalty = 0.0 if ok else min(1.0, (self.mfe_threshold - mfe) / 3.0)
        return FilterDecision(
            filter_name=FilterName.SECONDARY_STRUCTURE,
            passed=ok,
            value=mfe,
            threshold=self.mfe_threshold,
            reason=None if ok else (
                f"MFE={mfe:.1f} kcal/mol < threshold={self.mfe_threshold:.1f}"
            ),
            penalty=penalty,
        )

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _max_homopolymer(seq: str) -> int:
        """Find the longest homopolymer run in a sequence."""
        if not seq:
            return 0
        max_run = 1
        current_run = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i - 1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        return max_run

    @staticmethod
    def _max_run(seq: str, char: str) -> int:
        """Find the longest run of a specific character."""
        max_run = 0
        current_run = 0
        for c in seq:
            if c == char:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        return max_run

    @staticmethod
    def _dinucleotide_repeat_fraction(seq: str) -> float:
        """Fraction of the sequence covered by the most common dinucleotide.

        GCGCGCGC → GC appears 3 times → covers 6/8 = 0.75
        """
        if len(seq) < 4:
            return 0.0
        dinucs = [seq[i:i+2] for i in range(len(seq) - 1)]
        counts = Counter(dinucs)
        most_common_count = counts.most_common(1)[0][1]
        # Each dinucleotide covers 2 nt, but overlapping
        return most_common_count / len(dinucs)

    @staticmethod
    def _max_self_complement(seq: str) -> int:
        """Find the longest stretch where a subsequence is complementary
        to another subsequence (potential for self-folding / dimerization).

        Uses a simple O(n²) approach: for each pair of positions,
        check how long the complementary stretch extends.
        """
        complement = {"A": "T", "T": "A", "G": "C", "C": "G"}
        n = len(seq)
        max_len = 0

        for i in range(n):
            for j in range(i + 4, n):  # minimum gap of 4 for hairpin loop
                k = 0
                while (
                    i + k < j - k
                    and j + k < n  # Note: j scans forward, i scans forward
                    and complement.get(seq[i + k], "") == seq[j - k]
                ):
                    k += 1
                max_len = max(max_len, k)

        return max_len

    @staticmethod
    def _compute_mfe(spacer: str) -> Optional[float]:
        """Compute minimum free energy using ViennaRNA RNAfold.

        Returns None if ViennaRNA is not installed.
        """
        try:
            result = subprocess.run(
                ["RNAfold", "--noPS"],
                input=spacer,
                capture_output=True,
                text=True,
                timeout=10,
            )
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                mfe_str = lines[1].split("(")[-1].rstrip(")")
                return float(mfe_str.strip())
        except FileNotFoundError:
            logger.debug("ViennaRNA not installed, skipping MFE filter")
        except (subprocess.TimeoutExpired, ValueError, IndexError):
            logger.warning("RNAfold failed for spacer: %s", spacer)
        return None
