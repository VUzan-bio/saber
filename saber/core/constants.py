"""Biological constants for SABER.

Centralised to avoid magic numbers scattered across modules.
Sources cited inline.
"""

# ---------------------------------------------------------------------------
# Reference genome
# ---------------------------------------------------------------------------
H37RV_ACCESSION = "NC_000962.3"
H37RV_LENGTH = 4_411_532
H37RV_GC_CONTENT = 0.656                           # 65.6% GC

# ---------------------------------------------------------------------------
# Cas12a parameters
# ---------------------------------------------------------------------------
ASCAS12A_PAM = "TTTV"                               # V = A, C, or G
ENASCAS12A_PAMS = ("TTYN", "VTTV")                  # relaxed PAMs
SPACER_LENGTH_MIN = 20
SPACER_LENGTH_MAX = 24
SPACER_LENGTH_DEFAULT = 20
SEED_REGION_END = 8                                  # positions 1-8 from PAM-proximal

# ---------------------------------------------------------------------------
# Hard filter thresholds (Module 2)
# ---------------------------------------------------------------------------
GC_MIN = 0.40
GC_MAX = 0.60
HOMOPOLYMER_MAX = 4
MFE_THRESHOLD = -2.0                                # kcal/mol; more negative = worse

# ---------------------------------------------------------------------------
# Off-target screening (Module 3)
# ---------------------------------------------------------------------------
OFFTARGET_MISMATCH_THRESHOLD = 3                    # flag if ≤3 mismatches
BOWTIE2_SEED_LENGTH = 20
BOWTIE2_MAX_MISMATCHES = 3

# ---------------------------------------------------------------------------
# RPA primer constraints (Module 6)
# ---------------------------------------------------------------------------
RPA_PRIMER_LENGTH_MIN = 30
RPA_PRIMER_LENGTH_MAX = 35
RPA_TM_MIN = 60.0
RPA_TM_MAX = 65.0
RPA_AMPLICON_MIN = 100
RPA_AMPLICON_MAX = 200
PRIMER_DIMER_DG_THRESHOLD = -6.0                    # kcal/mol

# ---------------------------------------------------------------------------
# Heuristic scoring weights (Module 4, Level 1)
# Based on Kim et al. 2018 feature importance analysis
# ---------------------------------------------------------------------------
HEURISTIC_WEIGHTS = {
    "seed_position": 0.35,
    "gc": 0.20,
    "structure": 0.20,
    "homopolymer": 0.10,
    "offtarget": 0.15,
}

# ---------------------------------------------------------------------------
# Flanking sequence extraction
# ---------------------------------------------------------------------------
FLANKING_WINDOW = 500                                # bp upstream and downstream

# ---------------------------------------------------------------------------
# IUPAC degenerate bases
# ---------------------------------------------------------------------------
IUPAC_EXPAND: dict[str, set[str]] = {
    "A": {"A"}, "T": {"T"}, "G": {"G"}, "C": {"C"},
    "V": {"A", "C", "G"},
    "Y": {"C", "T"},
    "N": {"A", "T", "G", "C"},
}


def pam_matches(seq: str, pattern: str) -> bool:
    """Check if a 4-nt sequence matches a degenerate PAM pattern."""
    if len(seq) != len(pattern):
        return False
    return all(nt.upper() in IUPAC_EXPAND.get(p, set()) for nt, p in zip(seq, pattern))
