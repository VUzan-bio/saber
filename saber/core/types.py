"""Domain models for SABER.

Every module communicates through these types. They define the pipeline contract:
Target → Candidate → ScoredCandidate → MultiplexPanel.
"""

from __future__ import annotations

from dataclasses import field
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Strand(str, Enum):
    PLUS = "+"
    MINUS = "-"


class PAMVariant(str, Enum):
    """Cas12a PAM types."""
    TTTV = "TTTV"           # canonical AsCas12a
    TTYN = "TTYN"           # enAsCas12a relaxed
    VTTV = "VTTV"           # enAsCas12a relaxed


class Drug(str, Enum):
    ISONIAZID = "INH"
    RIFAMPICIN = "RIF"
    ETHAMBUTOL = "EMB"
    PYRAZINAMIDE = "PZA"
    FLUOROQUINOLONE = "FQ"
    AMINOGLYCOSIDE = "AG"
    BEDAQUILINE = "BDQ"
    LINEZOLID = "LZD"


class ValidationStatus(str, Enum):
    UNTESTED = "untested"
    IN_PROGRESS = "in_progress"
    VALIDATED = "validated"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Module 1: Target Definition
# ---------------------------------------------------------------------------

class Mutation(BaseModel):
    """A single resistance-conferring mutation from the WHO catalogue."""
    gene: str                                       # e.g. "rpoB"
    position: int                                   # codon number in gene coordinates
    ref_aa: str = Field(max_length=1)               # e.g. "S"
    alt_aa: str = Field(max_length=1)               # e.g. "L"
    nucleotide_change: Optional[str] = None         # e.g. "c.1349C>T"
    drug: Drug
    who_confidence: str = "assoc w resistance"      # WHO grading

    @property
    def label(self) -> str:
        return f"{self.gene}_{self.ref_aa}{self.position}{self.alt_aa}"


class Target(BaseModel):
    """A resolved genomic target — output of Module 1."""
    mutation: Mutation
    chrom: str = "NC_000962.3"                      # H37Rv accession
    genomic_pos: int                                 # 0-based position on genome
    ref_codon: str = Field(min_length=3, max_length=3)
    alt_codon: str = Field(min_length=3, max_length=3)
    flanking_seq: str                                # ±500 bp context window
    flanking_start: int                              # genome position of flanking_seq[0]

    @field_validator("ref_codon", "alt_codon")
    @classmethod
    def validate_codon(cls, v: str) -> str:
        if not set(v.upper()).issubset({"A", "T", "G", "C"}):
            raise ValueError(f"Invalid codon: {v}")
        return v.upper()

    @property
    def label(self) -> str:
        return self.mutation.label


# ---------------------------------------------------------------------------
# Module 2: Candidate Generation
# ---------------------------------------------------------------------------

class CrRNACandidate(BaseModel):
    """A single crRNA candidate — output of Module 2."""
    candidate_id: str                               # deterministic hash
    target_label: str                               # back-ref to Target
    spacer_seq: str = Field(min_length=20, max_length=24)
    pam_seq: str = Field(min_length=4, max_length=4)
    pam_variant: PAMVariant
    strand: Strand
    genomic_start: int                              # spacer start on genome
    genomic_end: int                                # spacer end on genome
    mutation_position_in_spacer: int                # 1-indexed from PAM-proximal end
    gc_content: float
    homopolymer_max: int
    mfe: Optional[float] = None                     # kcal/mol, from ViennaRNA

    @property
    def in_seed(self) -> bool:
        """Mutation falls in seed region (positions 1-8 from PAM)."""
        return self.mutation_position_in_spacer <= 8


class MismatchPair(BaseModel):
    """A WT/MUT spacer pair for discrimination analysis."""
    candidate_id: str
    wt_spacer: str
    mut_spacer: str
    mismatch_position: int                          # 1-indexed from PAM-proximal
    mismatch_type: str                              # e.g. "C>T"


# ---------------------------------------------------------------------------
# Module 3: Off-Target Screening
# ---------------------------------------------------------------------------

class OffTargetHit(BaseModel):
    """A single off-target alignment hit."""
    candidate_id: str
    hit_chrom: str
    hit_start: int
    hit_end: int
    mismatches: int
    alignment_score: float
    gene_annotation: Optional[str] = None           # gene overlapping the hit


class OffTargetReport(BaseModel):
    """Aggregated off-target results for one candidate."""
    candidate_id: str
    mtb_hits: list[OffTargetHit] = Field(default_factory=list)
    human_hits: list[OffTargetHit] = Field(default_factory=list)
    is_clean: bool = True                           # no hits with ≤3 mismatches

    @property
    def total_risky_hits(self) -> int:
        return len([h for h in self.mtb_hits + self.human_hits if h.mismatches <= 3])


# ---------------------------------------------------------------------------
# Module 4: Scoring
# ---------------------------------------------------------------------------

class HeuristicScore(BaseModel):
    """Rule-based score breakdown."""
    seed_position_score: float                      # closer to PAM = higher
    gc_penalty: float
    structure_penalty: float
    homopolymer_penalty: float
    offtarget_penalty: float
    composite: float                                # weighted sum


class MLScore(BaseModel):
    """ML-predicted efficiency."""
    model_name: str                                 # "seq_cnn" | "jepa_v1" | ...
    predicted_efficiency: float                     # 0-1 scale
    confidence: Optional[float] = None              # model uncertainty


class DiscriminationScore(BaseModel):
    """Predicted or measured discrimination between WT and MUT."""
    wt_activity: float
    mut_activity: float

    @property
    def ratio(self) -> float:
        if self.wt_activity == 0:
            return float("inf") if self.mut_activity > 0 else 0.0
        return self.mut_activity / self.wt_activity


class ScoredCandidate(BaseModel):
    """A candidate with all available scores — the unit for ranking."""
    candidate: CrRNACandidate
    offtarget: OffTargetReport
    heuristic: HeuristicScore
    ml_scores: list[MLScore] = Field(default_factory=list)
    discrimination: Optional[DiscriminationScore] = None
    rank: Optional[int] = None


# ---------------------------------------------------------------------------
# Module 5 & 6: Multiplex + Primers
# ---------------------------------------------------------------------------

class RPAPrimer(BaseModel):
    """An RPA primer."""
    seq: str = Field(min_length=28, max_length=38)
    tm: float
    direction: str = Field(pattern="^(fwd|rev)$")
    amplicon_start: int
    amplicon_end: int

    @property
    def amplicon_length(self) -> int:
        return self.amplicon_end - self.amplicon_start


class RPAPrimerPair(BaseModel):
    fwd: RPAPrimer
    rev: RPAPrimer
    dimer_dg: Optional[float] = None                # kcal/mol

    @property
    def amplicon_length(self) -> int:
        return self.fwd.amplicon_length


class PanelMember(BaseModel):
    """One slot in the multiplex panel."""
    target: Target
    selected_candidate: ScoredCandidate
    primers: RPAPrimerPair


class MultiplexPanel(BaseModel):
    """The final output — an N-plex diagnostic panel."""
    members: list[PanelMember]
    cross_reactivity_matrix: Optional[list[list[float]]] = None
    primer_dimer_matrix: Optional[list[list[float]]] = None
    panel_score: Optional[float] = None

    @property
    def plex(self) -> int:
        return len(self.members)


# ---------------------------------------------------------------------------
# Module 7: Experimental Validation
# ---------------------------------------------------------------------------

class ExperimentalResult(BaseModel):
    """A single wet-lab measurement."""
    candidate_id: str
    assay_type: str                                 # "fluorescence" | "electrochemical"
    target_concentration_nm: float
    signal_value: float
    signal_unit: str                                # "RFU" | "nA" | "pct_decrease"
    discrimination_ratio: Optional[float] = None
    notes: Optional[str] = None
    timestamp: Optional[str] = None                 # ISO 8601
