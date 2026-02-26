"""Domain models for SABER.

Every module communicates through these types. They define the pipeline contract:
  Mutation → Target → CrRNACandidate → ScoredCandidate → MultiplexPanel → ExperimentalResult

Design principles:
  - Pydantic BaseModel for validation + serialization
  - Enum-based taxonomies for type safety
  - All mutation types represented (SNP, indel, promoter, rRNA, large deletion)
  - Multi-organism support (not hardcoded to M.tb)
  - All Cas12a variants and PAM types
  - Flexible field constraints that accommodate non-standard mutation types
    (e.g. rRNA: 1-nt ref/alt, promoter: 1-nt, large deletion: "---")
  - DetectionStrategy enum for direct-overlap vs proximity-based detection
    (critical for PAM-desert regions like M.tb rpoB RRDR)

References:
  - WHO 2023 Catalogue of mutations in M. tuberculosis
  - Zetsche et al., Cell 2015 (Cas12a PAM)
  - Kleinstiver et al., Nature Biotech 2019 (enAsCas12a)
  - Kim et al., Nature Biotech 2018 (guide efficiency)
  - Li et al., Nature Biotech 2018 (DETECTR — proximity detection)
  - Broughton et al., Nature Biotech 2020 (SARS-CoV-2 DETECTR)
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ======================================================================
# Enums
# ======================================================================

class Strand(str, Enum):
    PLUS = "+"
    MINUS = "-"


class PAMVariant(str, Enum):
    """Cas12a PAM types across all supported variants.

    AsCas12a:       TTTV (canonical)
    enAsCas12a:     TTTV, TTTN, TTCN, TCTV, CTTV
    LbCas12a:       TTTV
    FnCas12a:       TTTV, KYTV
    Cas12a Ultra:   TTTV, TTTN, TTCN
    """
    TTTV = "TTTV"           # canonical AsCas12a / all variants
    TTTN = "TTTN"           # enAsCas12a relaxed 4th position
    TTCN = "TTCN"           # enAsCas12a C at position 3
    TCTV = "TCTV"           # enAsCas12a C at position 2
    CTTV = "CTTV"           # enAsCas12a C at position 1
    TTYN = "TTYN"           # legacy label (enAsCas12a)
    VTTV = "VTTV"           # legacy label (enAsCas12a)
    KYTV = "KYTV"           # FnCas12a relaxed


class Drug(str, Enum):
    """Antimicrobial drugs covered by SABER diagnostic panels.

    Covers first-line, second-line, and Group A/B/C agents
    per WHO 2022 treatment guidelines.
    """
    # First-line
    ISONIAZID = "INH"
    RIFAMPICIN = "RIF"
    ETHAMBUTOL = "EMB"
    PYRAZINAMIDE = "PZA"
    # Fluoroquinolones (Group A)
    FLUOROQUINOLONE = "FQ"
    LEVOFLOXACIN = "LFX"
    MOXIFLOXACIN = "MFX"
    # Group A
    BEDAQUILINE = "BDQ"
    LINEZOLID = "LZD"
    # Group B
    CLOFAZIMINE = "CFZ"
    CYCLOSERINE = "CS"
    # Group C / injectable
    AMINOGLYCOSIDE = "AG"
    AMIKACIN = "AMK"
    KANAMYCIN = "KAN"
    CAPREOMYCIN = "CAP"
    STREPTOMYCIN = "STR"
    # Newer agents
    DELAMANID = "DLM"
    PRETOMANID = "PMD"
    # Para-aminosalicylic acid
    PAS = "PAS"
    ETHIONAMIDE = "ETH"
    # Generic / multi-drug
    MULTI_DRUG = "MDR"
    OTHER = "OTHER"


class MutationCategory(str, Enum):
    """Biological classification of the mutation.

    Used by the resolver to dispatch to the correct resolution strategy,
    and by the filter to apply mutation-type-specific constraints.
    """
    AA_SUBSTITUTION = "aa_substitution"     # S450L, H445Y
    NUCLEOTIDE_SNP = "nucleotide_snp"       # c.1349C>T
    INSERTION = "insertion"                  # c.516_517insG
    DELETION = "deletion"                   # c.171delC
    LARGE_DELETION = "large_deletion"       # full gene deletion
    MNV = "mnv"                             # multi-nucleotide variant
    PROMOTER = "promoter"                   # inhA c.-15C>T
    RRNA = "rrna"                           # rrs A1401G
    FRAMESHIFT = "frameshift"               # pncA various
    INTERGENIC = "intergenic"               # mabA-inhA intergenic
    UNKNOWN = "unknown"


class DetectionStrategy(str, Enum):
    """How the crRNA contributes to mutation discrimination.

    DIRECT: The mutation lies within the spacer sequence. crRNA-level
        mismatch discrimination is the primary mechanism. The gold
        standard — mutation position in seed region gives best results.

    PROXIMITY: The mutation lies OUTSIDE the spacer but within the RPA
        amplicon. Detection relies on a combination of:
        (a) Allele-specific RPA primers bridging the mutation
        (b) crRNA trans-cleavage of the amplicon for signal generation
        (c) Optional: synthetic mismatches in crRNA to bias kinetics
        This strategy is the standard fallback for PAM-desert regions
        such as M.tb rpoB RRDR (70%+ GC, no T-rich PAM within 50 bp
        of codon 450). Many published DETECTR assays use this design.

    MISMATCH_ENHANCED: The crRNA is proximity-based but contains 1-2
        synthetic mismatches in the seed region to increase kinetic
        discrimination between WT and MUT amplicon conformations.
        Based on Chen et al. 2018 and Gootenberg et al. 2018.

    References:
        - Li et al., Nature Biotech 2018 (DETECTR)
        - Broughton et al., Nature Biotech 2020 (SARS-CoV-2 DETECTR)
        - Chen et al., Science 2018 (SHERLOCK — synthetic mismatches)
        - Gootenberg et al., Science 2018 (SHERLOCKv2)
    """
    DIRECT = "direct"
    PROXIMITY = "proximity"
    MISMATCH_ENHANCED = "mismatch_enhanced"


class ValidationStatus(str, Enum):
    """Experimental validation status for tracking."""
    UNTESTED = "untested"
    IN_PROGRESS = "in_progress"
    VALIDATED = "validated"
    FAILED = "failed"


class AssayType(str, Enum):
    """Supported detection modalities."""
    FLUORESCENCE = "fluorescence"           # standard CRISPR-Cas12a trans-cleavage
    ELECTROCHEMICAL = "electrochemical"     # electrochemical biosensor (CSEM)
    LATERAL_FLOW = "lateral_flow"           # lateral flow assay
    COLORIMETRIC = "colorimetric"           # colorimetric (AuNP)


class ScoringMode(str, Enum):
    """Available scoring backends."""
    HEURISTIC = "heuristic"
    SEQ_CNN = "seq_cnn"                     # Seq-deepCpf1 equivalent
    JEPA_EFFICIENCY = "jepa_efficiency"     # bDNA-JEPA Path A
    JEPA_DISCRIMINATION = "jepa_discrimination"  # bDNA-JEPA Path B
    JEPA_CONTEXT = "jepa_context"           # bDNA-JEPA Path C
    ENSEMBLE = "ensemble"


# ======================================================================
# Module 1: Target Definition
# ======================================================================

class Mutation(BaseModel):
    """A single resistance-conferring mutation from the WHO catalogue.

    Flexible enough for all mutation types:
    - AA substitution: gene=rpoB, position=450, ref_aa=S, alt_aa=L
    - rRNA: gene=rrs, position=1401, ref_aa=A, alt_aa=G (nt bases)
    - Promoter: gene=inhA, position=-15, ref_aa=C, alt_aa=T
    - Insertion: gene=pncA, position=NNN, nucleotide_change=c.NNNinsG
    - Deletion: gene=katG, position=NNN, nucleotide_change=c.NNN_NNNdel
    """
    gene: str                                       # e.g. "rpoB", "rrs", "inhA"
    position: int                                   # codon number OR nucleotide position
    ref_aa: str = Field(max_length=3)               # 1-char AA, or nt base for rRNA/promoter
    alt_aa: str = Field(max_length=3)               # 1-char AA, or nt base, or "*" for stop
    nucleotide_change: Optional[str] = None         # HGVS-like: "c.1349C>T", "c.-15C>T"
    drug: Drug = Drug.OTHER
    who_confidence: str = "assoc w resistance"      # WHO grading
    category: Optional[MutationCategory] = None     # auto-classified if None
    clinical_frequency: Optional[float] = None      # allele frequency in clinical isolates
    notes: Optional[str] = None

    @property
    def label(self) -> str:
        """Human-readable label: rpoB_S531L, rrs_A1401G, inhA_C-15T."""
        if self.position < 0:
            return f"{self.gene}_{self.ref_aa}{self.position}{self.alt_aa}"
        return f"{self.gene}_{self.ref_aa}{self.position}{self.alt_aa}"

    @property
    def is_promoter(self) -> bool:
        return self.position < 0

    @property
    def is_rrna(self) -> bool:
        return self.gene.lower() in ("rrs", "rrl", "rrf", "rrn")


class Target(BaseModel):
    """A resolved genomic target — output of Module 1.

    Flexible codon fields accommodate all mutation types:
    - AA substitution: ref_codon="TCG", alt_codon="TTG" (3 nt)
    - rRNA/promoter: ref_codon="A", alt_codon="G" (1 nt)
    - Large deletion: alt_codon="---"
    - Indel context: ref_codon=first 3 nt of context
    """
    mutation: Mutation
    chrom: str = "NC_000962.3"                      # reference accession
    genomic_pos: int                                 # 0-based position on genome
    ref_codon: str                                   # 1-3 nt, or "---" for deletion
    alt_codon: str                                   # 1-3 nt, or "---" for deletion
    flanking_seq: str                                # ±500 bp context window
    flanking_start: int                              # genome position of flanking_seq[0]

    @field_validator("ref_codon", "alt_codon")
    @classmethod
    def validate_codon(cls, v: str) -> str:
        """Accept standard codons, single nt (rRNA/promoter), and deletion markers."""
        if v == "---":
            return v
        if not v:
            raise ValueError("Codon cannot be empty")
        valid_chars = {"A", "T", "G", "C", "N", "-"}
        if not set(v.upper()).issubset(valid_chars):
            raise ValueError(f"Invalid codon: {v}")
        return v.upper()

    @property
    def label(self) -> str:
        return self.mutation.label

    @property
    def mutation_footprint_bp(self) -> int:
        """Size of the mutation footprint in base pairs.

        Codon SNP: 3, rRNA/promoter: 1, deletion marker: 3 (default).
        """
        if self.ref_codon == "---" or self.alt_codon == "---":
            return 3
        return len(self.ref_codon)


# ======================================================================
# Module 2: Candidate Generation
# ======================================================================

class CrRNACandidate(BaseModel):
    """A single crRNA candidate — output of Module 2 (PAM scanning).

    Spacer length is flexible (18-25 nt) to accommodate different
    Cas12a orthologs: AsCas12a (20), LbCas12a (20-23), FnCas12a (20).

    mutation_position_in_spacer:
      - For DIRECT strategy: 1-indexed position from PAM-proximal end
      - For PROXIMITY strategy: None (mutation outside spacer)
      - For large deletion presence/absence: None

    detection_strategy:
      - DIRECT: mutation inside spacer (gold standard)
      - PROXIMITY: nearest PAM, mutation outside spacer (PAM desert fallback)
      - MISMATCH_ENHANCED: proximity + synthetic mismatches

    proximity_distance:
      - Distance in bp from nearest spacer edge to mutation midpoint
      - 0 for DIRECT candidates (mutation overlaps spacer)
      - Positive integer for PROXIMITY (smaller = better)
      - Used for ranking proximity candidates and estimating RPA primer
        positioning requirements
    """
    candidate_id: str                               # deterministic hash
    target_label: str                               # back-ref to Target
    spacer_seq: str = Field(min_length=18, max_length=25)
    pam_seq: str = Field(min_length=4, max_length=4)
    pam_variant: PAMVariant
    strand: Strand
    genomic_start: int                              # spacer start on genome
    genomic_end: int                                # spacer end on genome
    mutation_position_in_spacer: Optional[int] = None  # 1-indexed from PAM-proximal
    gc_content: float
    homopolymer_max: int
    mfe: Optional[float] = None                     # kcal/mol, from ViennaRNA
    pam_activity_weight: float = 1.0                # relative PAM activity (0-1)
    detection_strategy: DetectionStrategy = DetectionStrategy.DIRECT
    proximity_distance: int = 0                     # bp from spacer edge to mutation, 0=direct

    @property
    def in_seed(self) -> bool:
        """Mutation falls in seed region (positions 1-8 from PAM).

        Returns False if mutation_position is None (e.g. proximity or large deletion).
        """
        if self.mutation_position_in_spacer is None:
            return False
        return 1 <= self.mutation_position_in_spacer <= 8

    @property
    def spacer_length(self) -> int:
        return len(self.spacer_seq)

    @property
    def is_direct(self) -> bool:
        """Candidate uses direct mismatch discrimination."""
        return self.detection_strategy == DetectionStrategy.DIRECT

    @property
    def is_proximity(self) -> bool:
        """Candidate requires RPA-level discrimination (proximity mode)."""
        return self.detection_strategy in (
            DetectionStrategy.PROXIMITY,
            DetectionStrategy.MISMATCH_ENHANCED,
        )


class MismatchPair(BaseModel):
    """A WT/MUT spacer pair for discrimination analysis.

    For each crRNA candidate targeting the mutant allele, we generate
    the corresponding wild-type spacer. The mismatch position and type
    determine predicted discrimination ratio.

    For indels, the WT spacer matches the reference (no indel) and the
    MUT spacer matches the mutant (with indel) — or vice versa.

    For PROXIMITY candidates, both spacers are identical (no mismatch
    in the crRNA), and discrimination comes from RPA primer design.
    """
    candidate_id: str
    wt_spacer: str
    mut_spacer: str
    mismatch_positions: list[int] = Field(default_factory=list)  # 1-indexed, multiple for MNV
    mismatch_type: str                              # e.g. "C>T", "ins_G", "del_3bp", "proximity"
    mutation_category: MutationCategory = MutationCategory.AA_SUBSTITUTION
    detection_strategy: DetectionStrategy = DetectionStrategy.DIRECT

    @property
    def mismatch_position(self) -> int:
        """Primary mismatch position (backward compat)."""
        return self.mismatch_positions[0] if self.mismatch_positions else 0

    @property
    def num_mismatches(self) -> int:
        return len(self.mismatch_positions)

    @property
    def is_proximity_pair(self) -> bool:
        """No crRNA-level mismatch; discrimination via RPA."""
        return self.detection_strategy != DetectionStrategy.DIRECT


# ======================================================================
# Module 3: Off-Target Screening
# ======================================================================

class OffTargetHit(BaseModel):
    """A single off-target alignment hit."""
    candidate_id: str
    hit_chrom: str
    hit_start: int
    hit_end: int
    mismatches: int
    alignment_score: float
    gene_annotation: Optional[str] = None           # gene overlapping the hit
    is_coding: bool = False                         # hit is in a coding region
    database: str = "mtb"                           # "mtb" | "human" | "cross_reactivity"


class OffTargetReport(BaseModel):
    """Aggregated off-target results for one candidate."""
    candidate_id: str
    mtb_hits: list[OffTargetHit] = Field(default_factory=list)
    human_hits: list[OffTargetHit] = Field(default_factory=list)
    cross_reactivity_hits: list[OffTargetHit] = Field(default_factory=list)
    is_clean: bool = True                           # no hits with ≤3 mismatches

    @property
    def total_risky_hits(self) -> int:
        all_hits = self.mtb_hits + self.human_hits + self.cross_reactivity_hits
        return len([h for h in all_hits if h.mismatches <= 3])

    @property
    def worst_mtb_mismatches(self) -> int:
        """Fewest mismatches among MTB off-target hits (lower = worse)."""
        if not self.mtb_hits:
            return 999
        return min(h.mismatches for h in self.mtb_hits)


# ======================================================================
# Module 4: Scoring
# ======================================================================

class HeuristicScore(BaseModel):
    """Rule-based score breakdown (Module 4 Level 1).

    Weighted sub-scores from Kim et al. 2018 feature analysis:
    - seed_position_score (0.35): closer to PAM = higher
    - gc_penalty (0.20): deviation from 50% GC
    - structure_penalty (0.20): MFE-based secondary structure
    - homopolymer_penalty (0.10): long homopolymer runs
    - offtarget_penalty (0.15): exponential decay with mismatch count

    For PROXIMITY candidates, the scoring formula shifts weights:
    - seed_position_score → replaced by proximity_score (closer = better)
    - pam_activity_weight is upweighted (since crRNA cuts amplicon, not allele)
    """
    seed_position_score: float                      # closer to PAM = higher
    gc_penalty: float
    structure_penalty: float
    homopolymer_penalty: float
    offtarget_penalty: float
    composite: float                                # weighted sum
    proximity_bonus: float = 0.0                    # bonus for nearby proximity (smaller dist)

    @property
    def breakdown(self) -> dict[str, float]:
        return {
            "seed": self.seed_position_score,
            "gc": self.gc_penalty,
            "structure": self.structure_penalty,
            "homopolymer": self.homopolymer_penalty,
            "offtarget": self.offtarget_penalty,
            "proximity_bonus": self.proximity_bonus,
            "composite": self.composite,
        }


class MLScore(BaseModel):
    """ML-predicted efficiency (Module 4 Level 2/3).

    Supports multiple model backends:
    - seq_cnn: Seq-deepCpf1 equivalent (Kim 2018 HT-PAMDA data)
    - jepa_efficiency: bDNA-JEPA Path A (direct regression)
    - jepa_discrimination: bDNA-JEPA Path B (pairwise WT/MUT)
    - jepa_context: bDNA-JEPA Path C (genomic context embeddings)
    """
    model_name: str                                 # ScoringMode value
    predicted_efficiency: float                     # 0-1 scale
    confidence: Optional[float] = None              # model uncertainty (epistemic)
    embedding: Optional[list[float]] = None         # latent representation (JEPA)
    training_cycle: Optional[int] = None            # active learning cycle number


class DiscriminationScore(BaseModel):
    """Predicted or measured discrimination between WT and MUT.

    The discrimination ratio is the key metric for diagnostic crRNA design:
    MUT_activity / WT_activity > 2.0 is the minimum threshold.
    Higher is better for reliable genotyping.

    For PROXIMITY candidates:
    - wt_activity and mut_activity are AMPLICON-LEVEL (RPA product)
    - Discrimination comes from allele-specific RPA, not crRNA mismatch
    - Still tracked through same interface for pipeline uniformity
    """
    wt_activity: float                              # trans-cleavage on WT target
    mut_activity: float                             # trans-cleavage on MUT target
    model_name: Optional[str] = None                # which model predicted this
    is_measured: bool = False                        # experimental vs predicted
    detection_strategy: DetectionStrategy = DetectionStrategy.DIRECT

    @property
    def ratio(self) -> float:
        """MUT/WT ratio. >2.0 = good discrimination."""
        if self.wt_activity == 0:
            return float("inf") if self.mut_activity > 0 else 0.0
        return self.mut_activity / self.wt_activity

    @property
    def passes_threshold(self) -> bool:
        """Meets minimum discrimination threshold (ratio ≥ 2.0)."""
        return self.ratio >= 2.0


class ScoredCandidate(BaseModel):
    """A candidate with all available scores — the unit for ranking.

    Combines heuristic, ML, and discrimination scores. The rank field
    is set during panel optimization.

    For PROXIMITY candidates, ranking accounts for:
    - Proximity distance (closer to mutation = higher rank)
    - PAM activity weight (higher = better trans-cleavage)
    - Spacer biophysical quality (GC, structure, off-targets)
    """
    candidate: CrRNACandidate
    offtarget: OffTargetReport
    heuristic: HeuristicScore
    ml_scores: list[MLScore] = Field(default_factory=list)
    discrimination: Optional[DiscriminationScore] = None
    rank: Optional[int] = None
    validation_status: ValidationStatus = ValidationStatus.UNTESTED

    @property
    def best_ml_score(self) -> Optional[float]:
        """Highest ML-predicted efficiency across all models."""
        if not self.ml_scores:
            return None
        return max(s.predicted_efficiency for s in self.ml_scores)

    @property
    def composite_score(self) -> float:
        """Best available composite score for ranking.

        Priority: ML ensemble > single ML > heuristic.
        Proximity candidates get a 0.8× penalty to prefer direct when both exist.
        """
        base = (
            max(s.predicted_efficiency for s in self.ml_scores)
            if self.ml_scores
            else self.heuristic.composite
        )
        if self.candidate.is_proximity:
            return base * 0.8
        return base

    @property
    def detection_strategy(self) -> DetectionStrategy:
        return self.candidate.detection_strategy


# ======================================================================
# Module 5 & 6: Multiplex + Primers
# ======================================================================

class RPAPrimer(BaseModel):
    """An RPA primer.

    RPA primers are 28-38 nt, Tm 60-65°C, and must avoid
    secondary structure and primer-dimer formation.

    For PROXIMITY candidates, one primer must be allele-specific
    (3' end overlaps the mutation site for discrimination).
    """
    seq: str = Field(min_length=28, max_length=38)
    tm: float
    direction: str = Field(pattern="^(fwd|rev)$")
    amplicon_start: int
    amplicon_end: int
    gc_content: Optional[float] = None
    is_allele_specific: bool = False                # 3' end overlaps mutation
    allele_specific_position: Optional[int] = None  # mutation position from 3' end

    @property
    def amplicon_length(self) -> int:
        return self.amplicon_end - self.amplicon_start

    @property
    def length(self) -> int:
        return len(self.seq)


class RPAPrimerPair(BaseModel):
    """A forward + reverse primer pair for RPA amplification.

    For PROXIMITY detection, at least one primer must be allele-specific.
    The amplicon must contain the crRNA target site.
    """
    fwd: RPAPrimer
    rev: RPAPrimer
    dimer_dg: Optional[float] = None                # kcal/mol, from Primer3 ntthal
    amplicon_seq: Optional[str] = None              # predicted amplicon sequence
    detection_strategy: DetectionStrategy = DetectionStrategy.DIRECT

    @property
    def amplicon_length(self) -> int:
        return self.rev.amplicon_end - self.fwd.amplicon_start

    @property
    def is_compatible(self) -> bool:
        """Dimer ΔG above threshold (-6.0 kcal/mol)."""
        if self.dimer_dg is None:
            return True
        return self.dimer_dg > -6.0

    @property
    def has_allele_specific_primer(self) -> bool:
        """At least one primer provides allele-specific amplification."""
        return self.fwd.is_allele_specific or self.rev.is_allele_specific


class PanelMember(BaseModel):
    """One slot in the multiplex panel."""
    target: Target
    selected_candidate: ScoredCandidate
    primers: Optional[RPAPrimerPair] = None         # None if primers not yet designed
    channel: Optional[str] = None                   # detection channel (FAM, HEX, etc.)

    @property
    def label(self) -> str:
        return self.target.label

    @property
    def is_complete(self) -> bool:
        """All components designed and validated."""
        return self.primers is not None

    @property
    def detection_strategy(self) -> DetectionStrategy:
        return self.selected_candidate.detection_strategy

    @property
    def requires_allele_specific_primers(self) -> bool:
        """Proximity candidates need AS-RPA primers for discrimination."""
        return self.selected_candidate.candidate.is_proximity


class MultiplexPanel(BaseModel):
    """The final output — an N-plex diagnostic panel.

    Contains the complete panel specification: targets, crRNAs, primers,
    and compatibility matrices for quality assessment.
    """
    members: list[PanelMember]
    cross_reactivity_matrix: Optional[list[list[float]]] = None
    primer_dimer_matrix: Optional[list[list[float]]] = None
    panel_score: Optional[float] = None
    optimizer_iterations: Optional[int] = None
    optimizer_temperature: Optional[float] = None

    @property
    def plex(self) -> int:
        return len(self.members)

    @property
    def targets(self) -> list[str]:
        return [m.label for m in self.members]

    @property
    def complete_members(self) -> int:
        return sum(1 for m in self.members if m.is_complete)

    @property
    def worst_cross_reactivity(self) -> Optional[float]:
        """Highest off-diagonal value in cross-reactivity matrix."""
        if self.cross_reactivity_matrix is None:
            return None
        worst = 0.0
        n = len(self.cross_reactivity_matrix)
        for i in range(n):
            for j in range(n):
                if i != j:
                    worst = max(worst, self.cross_reactivity_matrix[i][j])
        return worst

    @property
    def primer_conflicts(self) -> int:
        """Number of primer pairs with ΔG < -6.0 kcal/mol."""
        if self.primer_dimer_matrix is None:
            return 0
        count = 0
        n = len(self.primer_dimer_matrix)
        for i in range(n):
            for j in range(i + 1, n):
                if self.primer_dimer_matrix[i][j] < -6.0:
                    count += 1
        return count

    @property
    def direct_members(self) -> list[PanelMember]:
        """Panel members using direct mismatch detection."""
        return [m for m in self.members if m.detection_strategy == DetectionStrategy.DIRECT]

    @property
    def proximity_members(self) -> list[PanelMember]:
        """Panel members requiring proximity/AS-RPA detection."""
        return [m for m in self.members if m.detection_strategy != DetectionStrategy.DIRECT]


# ======================================================================
# Module 7: Experimental Validation
# ======================================================================

class ExperimentalConditions(BaseModel):
    """Standardised experimental conditions for reproducibility."""
    cas12a_variant: str = "enAsCas12a"
    cas12a_concentration_nm: float = 100.0
    crrna_concentration_nm: float = 200.0
    reporter_type: str = "FQ"                       # fluorophore-quencher
    reporter_concentration_nm: float = 500.0
    temperature_c: float = 37.0
    incubation_minutes: float = 60.0
    buffer: str = "NEBuffer 2.1"
    rpa_kit: Optional[str] = "TwistAmp Basic"
    uses_allele_specific_rpa: bool = False          # proximity mode


class ExperimentalResult(BaseModel):
    """A single wet-lab measurement.

    Tracks both fluorescence and electrochemical readouts,
    with full provenance for active learning data export.
    """
    candidate_id: str
    assay_type: AssayType
    target_concentration_nm: float
    signal_value: float
    signal_unit: str                                # "RFU", "nA", "pct_decrease"
    background_signal: Optional[float] = None       # no-target control
    discrimination_ratio: Optional[float] = None    # measured MUT/WT
    detection_strategy: DetectionStrategy = DetectionStrategy.DIRECT
    conditions: Optional[ExperimentalConditions] = None
    notes: Optional[str] = None
    timestamp: Optional[str] = None                 # ISO 8601
    batch_id: Optional[str] = None                  # experiment batch tracking
    operator: Optional[str] = None

    @property
    def signal_to_noise(self) -> Optional[float]:
        """Signal-to-noise ratio vs background."""
        if self.background_signal is None or self.background_signal == 0:
            return None
        return self.signal_value / self.background_signal

    @property
    def is_positive(self) -> bool:
        """Signal significantly above background (SNR > 3)."""
        snr = self.signal_to_noise
        if snr is None:
            return self.signal_value > 0
        return snr > 3.0


class ActiveLearningBatch(BaseModel):
    """A batch of candidates selected for experimental validation.

    Tracks the selection strategy used (top-K, uncertain, balanced)
    and links back to the training cycle for JEPA fine-tuning.
    """
    batch_id: str
    cycle_number: int                               # active learning cycle
    strategy: str                                   # "top_k", "uncertain", "balanced"
    candidates: list[str]                           # candidate_ids
    results: list[ExperimentalResult] = Field(default_factory=list)
    model_version: Optional[str] = None             # JEPA checkpoint used for selection
    notes: Optional[str] = None

    @property
    def num_tested(self) -> int:
        return len(self.results)

    @property
    def completion_rate(self) -> float:
        if not self.candidates:
            return 0.0
        return self.num_tested / len(self.candidates)
