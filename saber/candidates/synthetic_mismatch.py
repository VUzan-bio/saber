"""Synthetic mismatch enhancement for crRNA discrimination amplification.

The biological principle:
  - A crRNA targeting a resistance mutation has 0 mismatches vs mutant (MUT)
    and 1 natural mismatch vs wild-type (WT) at the mutation site.
  - This single mismatch often gives insufficient discrimination (2-5x) because
    Cas12a tolerates single mismatches, especially outside the seed region.
  - By introducing a DELIBERATE second mismatch near the mutation site, we create:
      crRNA vs MUT = 1 mismatch  (synthetic only — still active)
      crRNA vs WT  = 2 mismatches (synthetic + natural — activity collapses)
  - Discrimination ratio jumps from 2-5x to 10-100x.

This module systematically generates and ranks synthetic mismatch variants.

References:
  - Chen et al., Science 2018 (DETECTR) — mismatch-enhanced discrimination
  - Gootenberg et al., Science 2018 (SHERLOCKv2) — synthetic mismatch strategy
  - Teng et al., Genome Biology 2019 — position-dependent mismatch tolerance
  - Kim et al., Nature Methods 2020 — comprehensive Cas12a mismatch profiling
  - Broughton et al., Nature Biotechnology 2020 — DETECTR for SARS-CoV-2

Supports:
  - All Cas12a variants (AsCas12a, enAsCas12a, LbCas12a, FnCas12a)
  - All mutation types (SNP, rRNA, promoter, indel, MNV)
  - All bacterial organisms (mismatch tolerance is enzyme-specific, not organism-specific)
  - Both direct and proximity detection strategies
  - Multiple synthetic mismatch positions and substitution types
  - Thermodynamic mismatch penalty estimation
  - Combinatorial double-synthetic mismatches (optional, for difficult targets)
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

log = logging.getLogger(__name__)


# ======================================================================
# Constants: Cas12a mismatch tolerance landscape
# ======================================================================

class MismatchType(str, Enum):
    """RNA:DNA mismatch pair classification by thermodynamic stability.

    Ordered from MOST destabilising (best for synthetic mismatch) to
    LEAST destabilising (worst — Cas12a tolerates these, avoid them).

    The crRNA is RNA, the target strand is DNA.
    Notation: rX:dY means RNA base X paired with DNA base Y.
    """
    # Purine-purine: large, severely destabilising — BEST for synthetic MM
    rA_dA = "rA:dA"   # purine-purine clash
    rG_dA = "rG:dA"   # purine-purine clash
    rA_dG = "rA:dG"   # purine-purine clash
    rG_dG = "rG:dG"   # purine-purine clash

    # Pyrimidine-pyrimidine: small, moderately destabilising — GOOD
    rC_dC = "rC:dC"
    rU_dC = "rU:dC"
    rC_dT = "rC:dT"
    rU_dT = "rU:dT"   # pyrimidine-pyrimidine, weak

    # Wobble pairs: partially stable — AVOID for synthetic mismatch
    rG_dT = "rG:dT"   # G:T wobble, thermodynamically stable — WORST choice
    rU_dG = "rU:dG"   # U:G wobble — also bad

    # Purine-pyrimidine transversions: moderately destabilising — GOOD
    rA_dC = "rA:dC"   # purine vs pyrimidine transversion
    rC_dA = "rC:dA"   # pyrimidine vs purine transversion


# Penalty scores for each mismatch type (higher = more destabilising = better)
# Based on Kim et al. 2020 and Teng et al. 2019 mismatch profiling data
MISMATCH_DESTABILISATION: dict[MismatchType, float] = {
    # Purine-purine: most destabilising
    MismatchType.rA_dA: 1.00,
    MismatchType.rG_dA: 0.95,
    MismatchType.rA_dG: 0.90,
    MismatchType.rG_dG: 0.85,
    # Pyrimidine-pyrimidine: moderately destabilising
    MismatchType.rC_dC: 0.75,
    MismatchType.rU_dC: 0.70,
    MismatchType.rC_dT: 0.65,
    MismatchType.rU_dT: 0.55,
    # Wobble: tolerated — poor synthetic mismatch choices
    MismatchType.rG_dT: 0.20,  # G:T wobble — Cas12a tolerates this
    MismatchType.rU_dG: 0.25,  # U:G wobble — also tolerated
    # Purine-pyrimidine transversions: moderately destabilising
    MismatchType.rA_dC: 0.80,  # A:C transversion — good disruption
    MismatchType.rC_dA: 0.78,  # C:A transversion — good disruption
}

# Position-dependent mismatch tolerance for Cas12a (1-indexed from PAM)
# Values: fraction of activity LOST when a mismatch is at this position
# Higher = more sensitive to mismatch = better position for synthetic MM
# Based on comprehensive tiling studies: Kim et al. 2020, Strohkendl et al. 2018
#
# Seed region (1-8): very sensitive to mismatches
# Trunk region (9-14): moderate sensitivity
# Tail region (15-20+): low sensitivity — mismatches tolerated
_POSITION_SENSITIVITY_AsCas12a = {
    1: 0.95, 2: 0.93, 3: 0.90, 4: 0.88, 5: 0.85,
    6: 0.80, 7: 0.75, 8: 0.70,  # seed boundary
    9: 0.55, 10: 0.50, 11: 0.45, 12: 0.40,
    13: 0.35, 14: 0.30,  # trunk
    15: 0.22, 16: 0.18, 17: 0.15, 18: 0.12,
    19: 0.10, 20: 0.08, 21: 0.07, 22: 0.06, 23: 0.05,  # tail
}

_POSITION_SENSITIVITY_LbCas12a = {
    # LbCas12a has a wider sensitive region (positions 1-10)
    1: 0.97, 2: 0.95, 3: 0.93, 4: 0.90, 5: 0.88,
    6: 0.85, 7: 0.82, 8: 0.78, 9: 0.70, 10: 0.65,
    11: 0.50, 12: 0.42, 13: 0.35, 14: 0.28,
    15: 0.20, 16: 0.16, 17: 0.13, 18: 0.10,
    19: 0.08, 20: 0.06, 21: 0.05, 22: 0.04, 23: 0.03,
}

_POSITION_SENSITIVITY_enAsCas12a = {
    # enAsCas12a: similar to AsCas12a but slightly more tolerant overall
    # (engineered for broader PAM, slightly relaxed specificity)
    1: 0.93, 2: 0.90, 3: 0.87, 4: 0.84, 5: 0.80,
    6: 0.75, 7: 0.70, 8: 0.65,
    9: 0.50, 10: 0.45, 11: 0.40, 12: 0.35,
    13: 0.30, 14: 0.25,
    15: 0.20, 16: 0.16, 17: 0.13, 18: 0.10,
    19: 0.08, 20: 0.06, 21: 0.05, 22: 0.04, 23: 0.05,
}

_POSITION_SENSITIVITY_FnCas12a = {
    # FnCas12a: shorter seed (1-6), more tolerant outside seed
    1: 0.95, 2: 0.92, 3: 0.88, 4: 0.82, 5: 0.75, 6: 0.68,
    7: 0.45, 8: 0.38, 9: 0.32, 10: 0.28,
    11: 0.22, 12: 0.18, 13: 0.15, 14: 0.12,
    15: 0.10, 16: 0.08, 17: 0.06, 18: 0.05,
    19: 0.04, 20: 0.03,
}

POSITION_SENSITIVITY_PROFILES: dict[str, dict[int, float]] = {
    "AsCas12a": _POSITION_SENSITIVITY_AsCas12a,
    "enAsCas12a": _POSITION_SENSITIVITY_enAsCas12a,
    "LbCas12a": _POSITION_SENSITIVITY_LbCas12a,
    "FnCas12a": _POSITION_SENSITIVITY_FnCas12a,
    "Cas12a_ultra": _POSITION_SENSITIVITY_enAsCas12a,  # same profile
}

# Complementarity map: DNA base → Watson-Crick RNA complement
_DNA_TO_RNA_COMPLEMENT = {"A": "U", "T": "A", "C": "G", "G": "C"}
_RNA_BASES = ["A", "U", "C", "G"]
_DNA_BASES = ["A", "T", "C", "G"]


# ======================================================================
# Data models
# ======================================================================

@dataclass
class SyntheticMismatchSite:
    """A single synthetic mismatch introduced into a crRNA spacer.

    Attributes:
        position: 1-indexed position from PAM-proximal end of spacer
        original_rna_base: the RNA base in the unmodified crRNA at this position
        synthetic_rna_base: the new RNA base introduced as synthetic mismatch
        target_dna_base_wt: DNA base at this position in the WT target strand
        target_dna_base_mut: DNA base at this position in the MUT target strand
        mismatch_type_vs_wt: the RNA:DNA mismatch type when this crRNA faces WT
        mismatch_type_vs_mut: the RNA:DNA mismatch type when this crRNA faces MUT
        position_sensitivity: how sensitive Cas12a is to mismatches here (0-1)
        destabilisation_vs_wt: thermodynamic destabilisation score vs WT (0-1)
        destabilisation_vs_mut: thermodynamic destabilisation score vs MUT (0-1)
    """
    position: int
    original_rna_base: str
    synthetic_rna_base: str
    target_dna_base_wt: str
    target_dna_base_mut: str
    mismatch_type_vs_wt: Optional[MismatchType]
    mismatch_type_vs_mut: Optional[MismatchType]
    position_sensitivity: float
    destabilisation_vs_wt: float
    destabilisation_vs_mut: float


@dataclass
class NaturalMismatchSite:
    """The natural mismatch created by the resistance mutation.

    For a crRNA designed to match the MUT sequence:
      vs MUT: perfect match (0 mismatches at this site)
      vs WT:  1 mismatch (the resistance mutation site)
    """
    position: int  # 1-indexed from PAM-proximal end
    rna_base: str  # crRNA base (matches MUT)
    dna_base_wt: str  # WT DNA base (mismatches crRNA)
    dna_base_mut: str  # MUT DNA base (matches crRNA)
    mismatch_type_vs_wt: Optional[MismatchType]
    position_sensitivity: float
    destabilisation_vs_wt: float


@dataclass
class EnhancedVariant:
    """A crRNA variant with synthetic mismatch(es) added for discrimination.

    This is the core output of the synthetic mismatch module.
    """
    # Identity
    parent_candidate_id: str
    variant_id: str
    target_label: str

    # Sequences
    original_spacer_seq: str   # original crRNA spacer (DNA notation, matches MUT)
    enhanced_spacer_seq: str   # modified spacer with synthetic mismatch(es)
    wt_target_seq: str         # WT target sequence (for reference)
    mut_target_seq: str        # MUT target sequence (for reference)

    # Mismatch details
    natural_mismatch: NaturalMismatchSite
    synthetic_mismatches: list[SyntheticMismatchSite]

    # Discrimination prediction
    n_mismatches_vs_wt: int    # total mismatches when facing WT (natural + synthetic)
    n_mismatches_vs_mut: int   # total mismatches when facing MUT (synthetic only)
    predicted_activity_vs_mut: float   # predicted fraction of max activity vs MUT [0-1]
    predicted_activity_vs_wt: float    # predicted fraction of max activity vs WT [0-1]
    discrimination_score: float        # ratio: activity_mut / activity_wt (higher = better)
    composite_enhancement_score: float # combined score balancing activity + discrimination

    # Metadata
    detection_strategy: str    # "direct_enhanced" or "proximity_enhanced"
    cas_variant: str
    enhancement_type: str      # "single_synthetic" or "double_synthetic"

    # Provenance
    notes: list[str] = field(default_factory=list)


@dataclass
class EnhancementReport:
    """Summary report for all enhanced variants of a single candidate."""
    candidate_id: str
    target_label: str
    n_variants_generated: int
    n_variants_viable: int  # predicted activity vs MUT > threshold
    best_variant: Optional[EnhancedVariant]
    all_variants: list[EnhancedVariant]
    enhancement_possible: bool  # whether any variant improves discrimination
    natural_discrimination_score: float  # baseline without enhancement
    best_discrimination_score: float
    improvement_factor: float  # best / natural


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class EnhancementConfig:
    """Configuration for synthetic mismatch enhancement.

    Attributes:
        cas_variant: which Cas12a variant to use for position sensitivity
        min_activity_vs_mut: minimum predicted activity vs MUT to keep variant
            (too low = crRNA won't detect the mutant at all)
        max_activity_vs_wt: maximum predicted activity vs WT to consider
            (too high = still cleaves WT, poor discrimination)
        search_radius: how far from the natural mismatch to search for
            synthetic mismatch positions (in spacer positions)
        allow_seed_synthetic: whether to place synthetic mismatches in seed
            region (positions 1-8). Generally True — seed mismatches are
            most discriminating.
        allow_double_synthetic: whether to try 2 synthetic mismatches
            (3 total vs WT). More aggressive, sometimes needed for
            difficult targets.
        prefer_purine_purine: prioritize purine-purine mismatches (most
            destabilising). Almost always True.
        exclude_wobble_pairs: exclude G:T and U:G wobble pairs from
            synthetic mismatch options. Recommended True.
        activity_model: how to estimate activity. "heuristic" uses the
            position-sensitivity lookup. "jepa" uses the JEPA predictor
            (when available).
    """
    cas_variant: str = "enAsCas12a"
    min_activity_vs_mut: float = 0.30  # must retain ≥30% activity vs mutant
    max_activity_vs_wt: float = 0.15   # want ≤15% activity vs wild-type
    search_radius: int = 6             # search ±6 positions from mutation
    allow_seed_synthetic: bool = True
    allow_double_synthetic: bool = False
    prefer_purine_purine: bool = True
    exclude_wobble_pairs: bool = True
    activity_model: str = "heuristic"  # or "jepa" when available


# ======================================================================
# Core logic
# ======================================================================

def _rna_base_for_dna(dna_base: str) -> str:
    """Convert DNA target base to its Watson-Crick RNA complement."""
    return _DNA_TO_RNA_COMPLEMENT[dna_base.upper()]


def _classify_mismatch(rna_base: str, dna_base: str) -> Optional[MismatchType]:
    """Classify an RNA:DNA mismatch pair.

    Returns None if the pair is a Watson-Crick match (not a mismatch).
    """
    rna = rna_base.upper()
    dna = dna_base.upper()

    # Check if it's a match
    if _DNA_TO_RNA_COMPLEMENT.get(dna) == rna:
        return None  # Watson-Crick match, not a mismatch

    # Build the enum key
    rna_char = rna if rna != "T" else "U"  # normalise T→U for RNA
    dna_char = dna
    key = f"r{rna_char}:d{dna_char}"

    # Look up
    for mt in MismatchType:
        if mt.value == key:
            return mt

    # Fallback: shouldn't happen with valid bases
    log.warning(f"Unknown mismatch pair: r{rna_char}:d{dna_char}")
    return None


def _get_destabilisation(mm_type: Optional[MismatchType]) -> float:
    """Get destabilisation score for a mismatch type.

    Returns 0.0 for Watson-Crick matches (no destabilisation).
    """
    if mm_type is None:
        return 0.0
    return MISMATCH_DESTABILISATION.get(mm_type, 0.5)


def _get_position_sensitivity(
    position: int,
    cas_variant: str,
    spacer_length: int,
) -> float:
    """Get position-dependent mismatch sensitivity.

    Falls back to linear interpolation for positions beyond the lookup table.
    """
    profile = POSITION_SENSITIVITY_PROFILES.get(
        cas_variant,
        _POSITION_SENSITIVITY_enAsCas12a,  # default
    )
    if position in profile:
        return profile[position]

    # Extrapolate for positions beyond table (long spacers)
    max_pos = max(profile.keys())
    if position > max_pos:
        # Linear decay from last known value
        last_val = profile[max_pos]
        return max(0.02, last_val - 0.02 * (position - max_pos))

    return 0.05  # fallback


def _predict_activity(
    mismatches: list[tuple[int, float, float]],  # [(position, sensitivity, destabilisation)]
    cas_variant: str,
) -> float:
    """Predict Cas12a activity given a set of mismatches.

    Uses a cooperative mismatch model: individual mismatches reduce activity
    multiplicatively, but NEARBY mismatches (within 4 positions) have a
    cooperative penalty that makes the combined effect super-multiplicative.

    This captures the biological reality that two mismatches in the seed
    region destabilise the R-loop cooperatively — the DNA:RNA hybrid
    unwinds from the PAM-proximal end, and closely spaced mismatches
    create a longer destabilised stretch that collapses the R-loop entirely.

    Cooperativity model (based on Strohkendl et al. 2018, Kim et al. 2020):
      - Single mismatch: activity *= (1 - sensitivity × destabilisation)
      - Each pair of mismatches within 4 positions of each other:
        additional penalty of (1 - cooperativity_factor)
      - Cooperativity factor scales with proximity: adjacent = 0.60,
        2 apart = 0.40, 3 apart = 0.25, 4 apart = 0.15

    This is a heuristic baseline. The JEPA model (when plugged in) replaces
    this with learned predictions from actual mismatch profiling data, which
    captures the full cooperative landscape including position-specific and
    sequence-context effects.

    Returns:
        Predicted fraction of maximum activity [0.0, 1.0].
    """
    if not mismatches:
        return 1.0  # no mismatches = full activity

    # Step 1: Independent multiplicative effect
    activity = 1.0
    for _pos, sensitivity, destab in mismatches:
        reduction = sensitivity * destab
        activity *= (1.0 - min(reduction, 0.98))

    # Step 2: Cooperative penalty for nearby mismatch pairs
    # This is what makes synthetic mismatches actually work for discrimination
    COOPERATIVITY = {1: 0.60, 2: 0.40, 3: 0.25, 4: 0.15}

    if len(mismatches) >= 2:
        positions = sorted(m[0] for m in mismatches)
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = abs(positions[j] - positions[i])
                if distance in COOPERATIVITY:
                    coop_penalty = COOPERATIVITY[distance]
                    # Scale by average sensitivity of the two positions
                    avg_sens = (mismatches[i][1] + mismatches[j][1]) / 2
                    activity *= (1.0 - coop_penalty * avg_sens)

    return max(activity, 0.0)


def _generate_synthetic_sites(
    spacer_seq: str,
    wt_target_seq: str,
    mut_target_seq: str,
    natural_mm_position: int,
    config: EnhancementConfig,
) -> list[SyntheticMismatchSite]:
    """Generate all possible single synthetic mismatch sites.

    For each position within search_radius of the natural mismatch:
      - Skip the natural mismatch position itself
      - For each possible RNA base substitution:
        - Classify the mismatch vs WT and vs MUT
        - Score destabilisation
        - Filter out wobble pairs if configured

    Args:
        spacer_seq: crRNA spacer in DNA notation (5'→3', matches MUT target)
        wt_target_seq: WT target strand aligned to spacer (5'→3')
        mut_target_seq: MUT target strand aligned to spacer (5'→3')
        natural_mm_position: 1-indexed position of natural mismatch in spacer
        config: enhancement configuration

    Returns:
        List of SyntheticMismatchSite objects, unsorted.
    """
    spacer_len = len(spacer_seq)
    sites: list[SyntheticMismatchSite] = []

    # Search window around natural mismatch
    search_start = max(1, natural_mm_position - config.search_radius)
    search_end = min(spacer_len, natural_mm_position + config.search_radius)

    for pos in range(search_start, search_end + 1):
        # Skip the natural mismatch position — already has a mismatch vs WT
        if pos == natural_mm_position:
            continue

        # Skip if not in seed and seed-only mode
        # (We generally want seed positions, but don't restrict by default)

        idx = pos - 1  # 0-indexed into sequence strings

        # Current bases
        original_dna = spacer_seq[idx].upper()  # matches MUT
        original_rna = _rna_base_for_dna(
            # The crRNA complement: spacer is written as DNA target-sense,
            # so the crRNA base is the complement of the spacer base
            # Wait — this needs clarification.
            #
            # Convention in SABER:
            #   spacer_seq is the TARGET-SENSE DNA sequence (same as MUT target)
            #   The actual crRNA sequence is the COMPLEMENT of spacer_seq (in RNA)
            #   i.e., crRNA_base = RNA_complement_of(spacer_DNA_base)
            #
            # For the crRNA to match MUT perfectly:
            #   crRNA[i] = RNA_complement(MUT_target[i])
            #
            # Since spacer_seq == MUT_target (by SABER convention):
            #   crRNA[i] = RNA_complement(spacer_seq[i])
            original_dna  # this is the MUT target base, complement gives crRNA base
        )

        wt_dna = wt_target_seq[idx].upper() if idx < len(wt_target_seq) else original_dna
        mut_dna = mut_target_seq[idx].upper() if idx < len(mut_target_seq) else original_dna

        # At this position (NOT the mutation site):
        # WT and MUT should have the same DNA base (only mutation site differs)
        # original_rna is the crRNA base that matches MUT perfectly here

        # Position sensitivity
        pos_sensitivity = _get_position_sensitivity(pos, config.cas_variant, spacer_len)

        # Try each possible RNA substitution
        for new_rna in _RNA_BASES:
            if new_rna == original_rna:
                continue  # same base = no change

            # Classify mismatch when this crRNA faces WT target
            mm_vs_wt = _classify_mismatch(new_rna, wt_dna)
            # Classify mismatch when this crRNA faces MUT target
            mm_vs_mut = _classify_mismatch(new_rna, mut_dna)

            # The synthetic mismatch should create a mismatch vs BOTH targets
            # (since we're changing a non-mutation position where WT == MUT)
            # BUT: vs MUT it should be less severe than vs WT
            # Ideally: high destab vs WT, moderate destab vs MUT

            # If this substitution happens to be a Watson-Crick match with WT,
            # skip it — it would REDUCE discrimination (now matches WT better)
            if mm_vs_wt is None:
                continue  # matches WT = bad, skip

            # Filter wobble pairs if configured
            if config.exclude_wobble_pairs and mm_vs_wt in (
                MismatchType.rG_dT, MismatchType.rU_dG
            ):
                continue

            destab_wt = _get_destabilisation(mm_vs_wt)
            destab_mut = _get_destabilisation(mm_vs_mut)

            site = SyntheticMismatchSite(
                position=pos,
                original_rna_base=original_rna,
                synthetic_rna_base=new_rna,
                target_dna_base_wt=wt_dna,
                target_dna_base_mut=mut_dna,
                mismatch_type_vs_wt=mm_vs_wt,
                mismatch_type_vs_mut=mm_vs_mut,
                position_sensitivity=pos_sensitivity,
                destabilisation_vs_wt=destab_wt,
                destabilisation_vs_mut=destab_mut,
            )
            sites.append(site)

    return sites


def _build_enhanced_spacer(
    original_spacer: str,
    synthetic_sites: list[SyntheticMismatchSite],
) -> str:
    """Build the enhanced spacer sequence with synthetic mismatches applied.

    The spacer is in DNA target-sense notation. To introduce a synthetic
    mismatch, we need to change the spacer base such that the crRNA
    (which is the RNA complement of the spacer) has the desired synthetic base.

    crRNA_base = RNA_complement(spacer_DNA_base)
    → spacer_DNA_base = DNA_complement(crRNA_RNA_base)

    So if we want crRNA to have base X (RNA), the spacer should have
    the DNA base whose RNA complement is X.
    """
    spacer_list = list(original_spacer)

    rna_to_dna_spacer = {"A": "T", "U": "A", "C": "G", "G": "C"}

    for site in synthetic_sites:
        idx = site.position - 1
        # Convert desired RNA base to spacer DNA base
        new_spacer_base = rna_to_dna_spacer.get(
            site.synthetic_rna_base,
            site.synthetic_rna_base,  # fallback
        )
        spacer_list[idx] = new_spacer_base

    return "".join(spacer_list)


def _compute_natural_mismatch(
    spacer_seq: str,
    wt_target_seq: str,
    mut_target_seq: str,
    natural_mm_position: int,
    cas_variant: str,
    spacer_length: int,
) -> NaturalMismatchSite:
    """Characterise the natural mismatch at the mutation site."""
    idx = natural_mm_position - 1

    mut_dna = mut_target_seq[idx].upper()
    wt_dna = wt_target_seq[idx].upper()
    crna_rna = _rna_base_for_dna(mut_dna)  # crRNA matches MUT

    mm_type = _classify_mismatch(crna_rna, wt_dna)
    pos_sens = _get_position_sensitivity(natural_mm_position, cas_variant, spacer_length)
    destab = _get_destabilisation(mm_type)

    return NaturalMismatchSite(
        position=natural_mm_position,
        rna_base=crna_rna,
        dna_base_wt=wt_dna,
        dna_base_mut=mut_dna,
        mismatch_type_vs_wt=mm_type,
        position_sensitivity=pos_sens,
        destabilisation_vs_wt=destab,
    )


def _score_variant(
    natural: NaturalMismatchSite,
    synthetics: list[SyntheticMismatchSite],
    cas_variant: str,
) -> tuple[float, float, float, float]:
    """Score a variant's predicted activity and discrimination.

    Returns:
        (activity_vs_mut, activity_vs_wt, discrimination_score, composite_score)
    """
    # --- Activity vs MUT ---
    # Natural mismatch: 0 (crRNA matches MUT at mutation site)
    # Synthetic mismatches: each has destabilisation_vs_mut
    mm_vs_mut = [
        (s.position, s.position_sensitivity, s.destabilisation_vs_mut)
        for s in synthetics
        if s.mismatch_type_vs_mut is not None  # actual mismatch vs MUT
    ]
    activity_mut = _predict_activity(mm_vs_mut, cas_variant)

    # --- Activity vs WT ---
    # Natural mismatch at mutation site
    mm_vs_wt = [
        (natural.position, natural.position_sensitivity, natural.destabilisation_vs_wt)
    ]
    # Plus synthetic mismatches vs WT
    for s in synthetics:
        if s.mismatch_type_vs_wt is not None:
            mm_vs_wt.append(
                (s.position, s.position_sensitivity, s.destabilisation_vs_wt)
            )
    activity_wt = _predict_activity(mm_vs_wt, cas_variant)

    # --- Discrimination ---
    if activity_wt < 1e-6:
        discrimination = activity_mut / 1e-6  # cap at very high
    else:
        discrimination = activity_mut / activity_wt

    # --- Composite ---
    # Balance: high discrimination + acceptable activity vs MUT
    # Penalise if activity vs MUT is too low (can't detect mutant)
    # Penalise if activity vs WT is too high (still cleaves WT)
    composite = (
        0.5 * min(discrimination / 20.0, 1.0)  # normalise discrimination to ~20x max
        + 0.3 * activity_mut                     # reward MUT activity
        + 0.2 * (1.0 - activity_wt)             # reward low WT activity
    )

    return activity_mut, activity_wt, discrimination, composite


# ======================================================================
# Public API
# ======================================================================

def generate_enhanced_variants(
    candidate_id: str,
    target_label: str,
    spacer_seq: str,
    wt_target_seq: str,
    mut_target_seq: str,
    natural_mm_position: int,
    config: Optional[EnhancementConfig] = None,
    detection_strategy: str = "direct",
) -> EnhancementReport:
    """Generate all synthetic mismatch variants for a single crRNA candidate.

    This is the main entry point for the enhancement module.

    Args:
        candidate_id: ID of the parent CrRNACandidate
        target_label: e.g., "rpoB_S531L"
        spacer_seq: original spacer (DNA target-sense, matches MUT)
        wt_target_seq: WT target sequence aligned to spacer
        mut_target_seq: MUT target sequence aligned to spacer
        natural_mm_position: 1-indexed position of resistance mutation in spacer
        config: enhancement configuration (defaults to enAsCas12a standard)
        detection_strategy: "direct" or "proximity"

    Returns:
        EnhancementReport with all viable variants ranked by discrimination.
    """
    if config is None:
        config = EnhancementConfig()

    spacer_len = len(spacer_seq)

    # Validate inputs
    if natural_mm_position < 1 or natural_mm_position > spacer_len:
        log.warning(
            f"Natural mismatch position {natural_mm_position} out of range "
            f"for spacer length {spacer_len} on {target_label}"
        )
        return EnhancementReport(
            candidate_id=candidate_id,
            target_label=target_label,
            n_variants_generated=0,
            n_variants_viable=0,
            best_variant=None,
            all_variants=[],
            enhancement_possible=False,
            natural_discrimination_score=1.0,
            best_discrimination_score=1.0,
            improvement_factor=1.0,
        )

    # Characterise the natural mismatch
    natural = _compute_natural_mismatch(
        spacer_seq, wt_target_seq, mut_target_seq,
        natural_mm_position, config.cas_variant, spacer_len,
    )

    # Baseline discrimination (no synthetic mismatch)
    baseline_act_mut = 1.0  # perfect match vs MUT
    baseline_act_wt = _predict_activity(
        [(natural.position, natural.position_sensitivity, natural.destabilisation_vs_wt)],
        config.cas_variant,
    )
    baseline_disc = baseline_act_mut / max(baseline_act_wt, 1e-6)

    # Generate all synthetic mismatch sites
    synthetic_sites = _generate_synthetic_sites(
        spacer_seq, wt_target_seq, mut_target_seq,
        natural_mm_position, config,
    )

    # Build single-synthetic variants
    variants: list[EnhancedVariant] = []
    variant_counter = 0

    for site in synthetic_sites:
        variant_counter += 1
        vid = f"{candidate_id}_SM{variant_counter:03d}"

        enhanced_spacer = _build_enhanced_spacer(spacer_seq, [site])

        act_mut, act_wt, disc, composite = _score_variant(
            natural, [site], config.cas_variant,
        )

        notes = []
        if act_mut < config.min_activity_vs_mut:
            notes.append(f"LOW_MUT_ACTIVITY ({act_mut:.2f} < {config.min_activity_vs_mut})")
        if site.mismatch_type_vs_wt and "rG:dT" in site.mismatch_type_vs_wt.value:
            notes.append("WOBBLE_PAIR_VS_WT")

        variant = EnhancedVariant(
            parent_candidate_id=candidate_id,
            variant_id=vid,
            target_label=target_label,
            original_spacer_seq=spacer_seq,
            enhanced_spacer_seq=enhanced_spacer,
            wt_target_seq=wt_target_seq,
            mut_target_seq=mut_target_seq,
            natural_mismatch=natural,
            synthetic_mismatches=[site],
            n_mismatches_vs_wt=2,  # natural + 1 synthetic
            n_mismatches_vs_mut=1 if site.mismatch_type_vs_mut is not None else 0,
            predicted_activity_vs_mut=act_mut,
            predicted_activity_vs_wt=act_wt,
            discrimination_score=disc,
            composite_enhancement_score=composite,
            detection_strategy=f"{detection_strategy}_enhanced",
            cas_variant=config.cas_variant,
            enhancement_type="single_synthetic",
            notes=notes,
        )
        variants.append(variant)

    # Optionally generate double-synthetic variants (2 synthetic + 1 natural = 3 vs WT)
    if config.allow_double_synthetic and len(synthetic_sites) >= 2:
        # Only try pairs of the top-scoring single sites to limit combinatorics
        top_singles = sorted(
            synthetic_sites,
            key=lambda s: s.destabilisation_vs_wt * s.position_sensitivity,
            reverse=True,
        )[:8]  # top 8 → up to 28 pairs

        for s1, s2 in itertools.combinations(top_singles, 2):
            if s1.position == s2.position:
                continue

            variant_counter += 1
            vid = f"{candidate_id}_DM{variant_counter:03d}"

            enhanced_spacer = _build_enhanced_spacer(spacer_seq, [s1, s2])

            act_mut, act_wt, disc, composite = _score_variant(
                natural, [s1, s2], config.cas_variant,
            )

            notes = []
            if act_mut < config.min_activity_vs_mut:
                notes.append(f"LOW_MUT_ACTIVITY ({act_mut:.2f})")

            variant = EnhancedVariant(
                parent_candidate_id=candidate_id,
                variant_id=vid,
                target_label=target_label,
                original_spacer_seq=spacer_seq,
                enhanced_spacer_seq=enhanced_spacer,
                wt_target_seq=wt_target_seq,
                mut_target_seq=mut_target_seq,
                natural_mismatch=natural,
                synthetic_mismatches=[s1, s2],
                n_mismatches_vs_wt=3,  # natural + 2 synthetic
                n_mismatches_vs_mut=sum(
                    1 for s in [s1, s2] if s.mismatch_type_vs_mut is not None
                ),
                predicted_activity_vs_mut=act_mut,
                predicted_activity_vs_wt=act_wt,
                discrimination_score=disc,
                composite_enhancement_score=composite,
                detection_strategy=f"{detection_strategy}_enhanced",
                cas_variant=config.cas_variant,
                enhancement_type="double_synthetic",
                notes=notes,
            )
            variants.append(variant)

    # Filter viable variants
    viable = [
        v for v in variants
        if v.predicted_activity_vs_mut >= config.min_activity_vs_mut
    ]

    # Sort by composite score (best first)
    viable.sort(key=lambda v: v.composite_enhancement_score, reverse=True)

    best = viable[0] if viable else None
    best_disc = best.discrimination_score if best else baseline_disc

    report = EnhancementReport(
        candidate_id=candidate_id,
        target_label=target_label,
        n_variants_generated=len(variants),
        n_variants_viable=len(viable),
        best_variant=best,
        all_variants=viable,
        enhancement_possible=bool(viable) and best_disc > baseline_disc * 1.5,
        natural_discrimination_score=baseline_disc,
        best_discrimination_score=best_disc,
        improvement_factor=best_disc / max(baseline_disc, 1e-6),
    )

    return report


def enhance_candidate_batch(
    candidates: list[dict],
    config: Optional[EnhancementConfig] = None,
) -> list[EnhancementReport]:
    """Enhance a batch of candidates.

    Each candidate dict must have:
        candidate_id, target_label, spacer_seq,
        wt_target_seq, mut_target_seq, natural_mm_position

    Optional: detection_strategy (default "direct")

    This is the batch interface for integration with the SABER pipeline.
    """
    if config is None:
        config = EnhancementConfig()

    reports: list[EnhancementReport] = []

    for cand in candidates:
        # Skip candidates without mutation position (proximity candidates
        # without known position need different handling)
        mm_pos = cand.get("natural_mm_position")
        if mm_pos is None or mm_pos < 1:
            log.debug(
                f"Skipping {cand.get('candidate_id', '?')}: "
                f"no natural mismatch position (proximity candidate?)"
            )
            continue

        report = generate_enhanced_variants(
            candidate_id=cand["candidate_id"],
            target_label=cand["target_label"],
            spacer_seq=cand["spacer_seq"],
            wt_target_seq=cand["wt_target_seq"],
            mut_target_seq=cand["mut_target_seq"],
            natural_mm_position=mm_pos,
            config=config,
            detection_strategy=cand.get("detection_strategy", "direct"),
        )
        reports.append(report)

        if report.enhancement_possible:
            best = report.best_variant
            log.info(
                f"Enhanced {cand['candidate_id']} for {cand['target_label']}: "
                f"discrimination {report.natural_discrimination_score:.1f}x → "
                f"{report.best_discrimination_score:.1f}x "
                f"({report.improvement_factor:.1f}x improvement), "
                f"activity vs MUT = {best.predicted_activity_vs_mut:.2f}"
            )
        else:
            log.debug(
                f"No enhancement for {cand['candidate_id']}: "
                f"baseline discrimination = {report.natural_discrimination_score:.1f}x"
            )

    n_enhanced = sum(1 for r in reports if r.enhancement_possible)
    log.info(
        f"Enhancement batch: {len(reports)} candidates processed, "
        f"{n_enhanced} enhanced ({n_enhanced/max(len(reports),1)*100:.0f}%)"
    )

    return reports


# ======================================================================
# Pipeline integration helpers
# ======================================================================

def enhance_from_scored_candidates(
    scored_candidates: list,  # list of ScoredCandidate objects
    mismatch_pairs: list,     # list of MismatchPair objects
    config: Optional[EnhancementConfig] = None,
) -> list[EnhancementReport]:
    """Integration point for the SABER pipeline.

    Takes ScoredCandidate and MismatchPair objects directly from the pipeline
    and produces EnhancementReports.

    Only processes direct candidates with known mutation positions and
    corresponding mismatch pairs.
    """
    if config is None:
        config = EnhancementConfig()

    # Index mismatch pairs by candidate_id
    pair_map: dict[str, object] = {}
    for pair in mismatch_pairs:
        cid = getattr(pair, "candidate_id", None)
        if cid:
            pair_map[cid] = pair

    batch: list[dict] = []

    for sc in scored_candidates:
        cand = sc.candidate  # CrRNACandidate
        cid = cand.candidate_id
        mm_pos = cand.mutation_position_in_spacer

        if mm_pos is None or mm_pos < 1:
            continue  # proximity candidate or no position info

        pair = pair_map.get(cid)
        if pair is None:
            continue  # no mismatch pair available

        wt_seq = getattr(pair, "wt_spacer", None) or getattr(pair, "wt_seq", "")
        mut_seq = getattr(pair, "mut_spacer", None) or getattr(pair, "mut_seq", "")

        if not wt_seq or not mut_seq:
            continue

        strategy = str(getattr(cand, "detection_strategy", "direct"))
        if "PROXIMITY" in strategy.upper():
            strategy = "proximity"
        else:
            strategy = "direct"

        batch.append({
            "candidate_id": cid,
            "target_label": cand.target_label,
            "spacer_seq": cand.spacer_seq,
            "wt_target_seq": wt_seq,
            "mut_target_seq": mut_seq,
            "natural_mm_position": mm_pos,
            "detection_strategy": strategy,
        })

    return enhance_candidate_batch(batch, config)
