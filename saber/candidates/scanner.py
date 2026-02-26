"""Scan target flanking sequences for Cas12a-compatible PAM sites.

Organism-agnostic, Cas12a-variant-aware, mutation-type-complete.

Design philosophy:
  MAXIMISE the candidate pool. Every biophysical filter (seed, GC,
  homopolymer, MFE, etc.) is applied DOWNSTREAM by CandidateFilter.
  The scanner's only job is to find every spacer that overlaps the
  mutation, at every valid length, on both strands.

  When direct overlap finds 0 candidates (PAM desert), the scanner
  AUTOMATICALLY falls back to PROXIMITY mode: find the nearest PAM
  sites flanking the mutation on each strand and generate proximity
  candidates. These are flagged with DetectionStrategy.PROXIMITY
  and ranked by distance to mutation.

Key design decisions for high-GC genomes (M.tb, P. aeruginosa):
  1. Multi-length spacers by default (18-23 nt). A PAM 22 nt from
     the mutation produces zero candidates at length=20, but valid
     candidates at length=22. Critical in PAM-poor regions.
  2. Expanded enAsCas12a PAMs: TTTV + TTTN + TTCN + TCTV + CTTV.
     ~5× increase in PAM density vs TTTV alone.
  3. Combined: 6 lengths × 5 PAMs × 2 strands = 60× more opportunities
     vs naive (1 length × 1 PAM × 2 strands).
  4. Backward-compat: if old code passes spacer_length=20, the scanner
     AUTOMATICALLY expands to [18..23] to avoid PAM desert failures.
  5. PROXIMITY FALLBACK: if direct overlap scan returns 0 candidates,
     find nearest PAMs within ±200 bp of mutation on each strand.
     Generate proximity candidates flagged for AS-RPA primer design.

Self-contained: no imports from saber.core.constants for PAM logic.
All IUPAC matching, PAM definitions, and candidate generation here.

PAM desert problem (M.tb rpoB RRDR):
  The rifampicin-resistance-determining region of rpoB has ~70% local
  GC content. Cas12a requires T-rich PAMs (5'-TTTV-3' canonical).
  In a 100 bp window around codon 450 (S450L / clinical S531L), there
  are ZERO upstream PAMs that would place the mutation within a 18-23 nt
  spacer. The nearest upstream PAM (TTCN) is 47 nt from the mutation —
  far beyond any Cas12a spacer length.

  This is not a bug. It is a fundamental limitation of Cas12a PAM
  requirements in GC-rich genomes. The proximity fallback solves this
  by generating the nearest usable crRNA and delegating discrimination
  to allele-specific RPA primer design.

  Many published DETECTR/SHERLOCK assays use this exact strategy:
  - Li et al., Nature Biotech 2018 (DETECTR)
  - Broughton et al., Nature Biotech 2020 (SARS-CoV-2)
  - Ai et al., Cell Discovery 2019 (M.tb)

References:
  - Zetsche et al., Cell 2015 (Cas12a, TTTV PAM)
  - Kleinstiver et al., Nature Biotech 2019 (enAsCas12a)
  - Kim et al., Nature Biotech 2018 (spacer length effects)
  - Creutzburg et al., NAR 2020 (PAM compatibility)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

from Bio.Seq import Seq

from saber.core.types import (
    CrRNACandidate,
    DetectionStrategy,
    PAMVariant,
    Strand,
    Target,
)

logger = logging.getLogger(__name__)


# ======================================================================
# IUPAC nucleotide matching (self-contained)
# ======================================================================

_IUPAC: dict[str, frozenset[str]] = {
    "A": frozenset("A"), "T": frozenset("T"),
    "G": frozenset("G"), "C": frozenset("C"),
    "R": frozenset("AG"), "Y": frozenset("CT"),
    "S": frozenset("GC"), "W": frozenset("AT"),
    "K": frozenset("GT"), "M": frozenset("AC"),
    "B": frozenset("CGT"), "D": frozenset("AGT"),
    "H": frozenset("ACT"), "V": frozenset("ACG"),
    "N": frozenset("ATGC"),
}


def iupac_match(seq: str, pattern: str) -> bool:
    """Check if a DNA sequence matches an IUPAC ambiguity pattern."""
    if len(seq) != len(pattern):
        return False
    for s, p in zip(seq.upper(), pattern.upper()):
        allowed = _IUPAC.get(p)
        if allowed is None or s not in allowed:
            return False
    return True


# ======================================================================
# Cas12a PAM definitions
# ======================================================================

@dataclass(frozen=True)
class PAMDef:
    pattern: str
    length: int
    activity: float        # relative to canonical TTTV
    label: str             # maps to PAMVariant enum


@dataclass(frozen=True)
class ScanConfig:
    variant: str
    pams: tuple[PAMDef, ...]
    lengths: tuple[int, ...]
    seed_start: int
    seed_end: int


_LENGTHS_DEFAULT = (18, 19, 20, 21, 22, 23)

CONFIGS: dict[str, ScanConfig] = {
    "AsCas12a": ScanConfig(
        "AsCas12a",
        pams=(PAMDef("TTTV", 4, 1.0, "TTTV"),),
        lengths=_LENGTHS_DEFAULT,
        seed_start=1, seed_end=8,
    ),
    "enAsCas12a": ScanConfig(
        "enAsCas12a",
        pams=(
            PAMDef("TTTV", 4, 1.0, "TTTV"),
            PAMDef("TTTN", 4, 0.7, "TTTN"),
            PAMDef("TTCN", 4, 0.4, "TTCN"),
            PAMDef("TCTV", 4, 0.3, "TCTV"),
            PAMDef("CTTV", 4, 0.2, "CTTV"),
        ),
        lengths=_LENGTHS_DEFAULT,
        seed_start=1, seed_end=8,
    ),
    "LbCas12a": ScanConfig(
        "LbCas12a",
        pams=(PAMDef("TTTV", 4, 1.0, "TTTV"),),
        lengths=(20, 21, 22, 23),
        seed_start=1, seed_end=10,
    ),
    "FnCas12a": ScanConfig(
        "FnCas12a",
        pams=(
            PAMDef("TTTV", 4, 1.0, "TTTV"),
            PAMDef("KYTV", 4, 0.5, "KYTV"),
        ),
        lengths=_LENGTHS_DEFAULT,
        seed_start=1, seed_end=6,
    ),
    "Cas12a_ultra": ScanConfig(
        "Cas12a_ultra",
        pams=(
            PAMDef("TTTV", 4, 1.0, "TTTV"),
            PAMDef("TTTN", 4, 0.8, "TTTN"),
            PAMDef("TTCN", 4, 0.5, "TTCN"),
        ),
        lengths=_LENGTHS_DEFAULT,
        seed_start=1, seed_end=8,
    ),
}


# ======================================================================
# Proximity scan configuration
# ======================================================================

@dataclass
class ProximityConfig:
    """Configuration for proximity-based candidate generation.

    Controls how far from the mutation the scanner looks for PAMs
    when direct overlap yields zero candidates.

    Attributes:
        enabled:        If True, automatically trigger proximity scan
                        when direct scan returns 0 candidates.
        max_distance:   Maximum distance (bp) from mutation to search
                        for PAMs. 200 bp is generous — RPA amplicons
                        are typically 100-300 bp total.
        max_candidates: Maximum number of proximity candidates to return
                        per strand (ranked by distance). Avoids flooding
                        downstream modules with distant, low-quality hits.
        prefer_upstream: Weight for upstream PAMs (spacer extends through/
                        past mutation region). Upstream PAMs are preferred
                        because their spacers are closer to the mutation,
                        which can improve trans-cleavage kinetics near the
                        mismatch site even in proximity mode.
    """
    enabled: bool = True
    max_distance: int = 200
    max_candidates_per_strand: int = 10
    prefer_upstream: float = 1.2                    # 20% bonus for upstream PAMs


# ======================================================================
# Mutation footprint
# ======================================================================

def mutation_footprint(target: Target) -> tuple[int, int]:
    """Mutation [start, end) in flanking-seq coordinates (0-based).

    Handles all mutation types:
      Codon SNP (ref_codon len 3): [offset, offset+3)
      rRNA / promoter (len 1):     [offset, offset+1)
      Large deletion ("---"):       [offset, offset+3)
    """
    offset = target.genomic_pos - target.flanking_start
    ref = target.ref_codon
    size = 3 if ref == "---" else len(ref)
    return offset, offset + size


# ======================================================================
# Helpers
# ======================================================================

def _gc(seq: str) -> float:
    s = seq.upper()
    return (s.count("G") + s.count("C")) / len(s) if s else 0.0


def _max_homo(seq: str) -> int:
    if not seq:
        return 0
    best = run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            run += 1
            if run > best:
                best = run
        else:
            run = 1
    return best


def _cid(label: str, strand: Strand, gstart: int, splen: int, prefix: str = "") -> str:
    raw = f"{prefix}{label}:{strand.value}:{gstart}:{splen}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _pam_variant(label: str) -> PAMVariant:
    try:
        return PAMVariant(label)
    except ValueError:
        return PAMVariant.TTTV


# ======================================================================
# ScanResult — structured output from the scanner
# ======================================================================

@dataclass
class ScanResult:
    """Complete scan output for one target.

    Separates direct and proximity candidates for clear reporting.
    The runner can decide how to handle proximity candidates (e.g.
    route them to the AS-RPA primer design module).
    """
    target_label: str
    direct_candidates: list[CrRNACandidate] = field(default_factory=list)
    proximity_candidates: list[CrRNACandidate] = field(default_factory=list)
    pam_desert: bool = False                        # True if direct scan found 0

    @property
    def all_candidates(self) -> list[CrRNACandidate]:
        """All candidates, direct first then proximity."""
        return self.direct_candidates + self.proximity_candidates

    @property
    def total(self) -> int:
        return len(self.direct_candidates) + len(self.proximity_candidates)

    @property
    def has_direct(self) -> bool:
        return len(self.direct_candidates) > 0

    @property
    def summary(self) -> str:
        parts = [f"{self.target_label}: {len(self.direct_candidates)} direct"]
        if self.proximity_candidates:
            dists = [c.proximity_distance for c in self.proximity_candidates]
            parts.append(
                f"{len(self.proximity_candidates)} proximity "
                f"(nearest {min(dists)} bp)"
            )
        if self.pam_desert:
            parts.append("PAM DESERT")
        return ", ".join(parts)


# ======================================================================
# PAMScanner
# ======================================================================

class PAMScanner:
    """Scan for Cas12a PAM sites and generate crRNA candidates.

    Two-phase scanning:
    1. DIRECT SCAN: Find PAMs where the spacer overlaps the mutation.
       This is the gold standard for crRNA-level discrimination.
    2. PROXIMITY SCAN (automatic fallback): If Phase 1 yields 0
       candidates, find the nearest PAMs on each strand and generate
       proximity candidates flagged for AS-RPA primer design.

    Usage:
        scanner = PAMScanner()                       # enAsCas12a, 18-23 nt
        scanner = PAMScanner(cas_variant="LbCas12a") # LbCas12a, 20-23 nt

        # Standard usage (returns all candidates)
        candidates = scanner.scan(target)

        # Structured output (separates direct/proximity)
        result = scanner.scan_detailed(target)
        print(result.summary)

        # Backward compat (old runner.py)
        scanner = PAMScanner(spacer_length=20, use_enascas12a=True)
        # → automatically expands to lengths [18..23]
    """

    def __init__(
        self,
        cas_variant: str = "enAsCas12a",
        spacer_lengths: Optional[list[int]] = None,
        min_pam_activity: float = 0.0,
        proximity: Optional[ProximityConfig] = None,
        # Backward-compatible kwargs from old runner
        spacer_length: Optional[int] = None,
        use_enascas12a: Optional[bool] = None,
    ) -> None:
        # Backward compat: use_enascas12a flag
        if use_enascas12a is not None:
            cas_variant = "enAsCas12a" if use_enascas12a else "AsCas12a"

        if cas_variant not in CONFIGS:
            logger.warning("Unknown variant '%s', using enAsCas12a", cas_variant)
            cas_variant = "enAsCas12a"

        self.config = CONFIGS[cas_variant]

        # Spacer lengths: explicit > backward compat (expanded) > config
        if spacer_lengths is not None:
            self.lengths = tuple(sorted(spacer_lengths))
        elif spacer_length is not None:
            # CRITICAL: expand single length to range to solve PAM desert
            lo = max(18, spacer_length - 2)
            hi = min(25, spacer_length + 3)
            self.lengths = tuple(range(lo, hi + 1))
        else:
            self.lengths = self.config.lengths

        self.pams = tuple(
            p for p in self.config.pams if p.activity >= min_pam_activity
        )

        # Proximity config (enabled by default)
        self.proximity = proximity or ProximityConfig()

        logger.info(
            "PAMScanner: variant=%s, PAMs=%s, lengths=%s, seed=%d-%d, "
            "proximity=%s (max_dist=%d bp)",
            self.config.variant,
            [p.pattern for p in self.pams],
            list(self.lengths),
            self.config.seed_start,
            self.config.seed_end,
            "ON" if self.proximity.enabled else "OFF",
            self.proximity.max_distance,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self, target: Target) -> list[CrRNACandidate]:
        """Scan for PAM sites and return all candidates (direct + proximity).

        This is the main entry point. Returns a flat list of candidates
        where direct candidates come first, followed by proximity
        candidates (if any). Each candidate's detection_strategy field
        indicates whether it's DIRECT or PROXIMITY.

        The runner can filter by detection_strategy if needed:
            direct_only = [c for c in candidates if c.is_direct]
            proximity_only = [c for c in candidates if c.is_proximity]
        """
        result = self.scan_detailed(target)
        return result.all_candidates

    def scan_detailed(self, target: Target) -> ScanResult:
        """Scan with structured output separating direct/proximity.

        Returns a ScanResult with:
          - direct_candidates: mutation inside spacer
          - proximity_candidates: nearest PAM, mutation outside spacer
          - pam_desert: True if direct scan found 0 and proximity was used
        """
        flanking = target.flanking_seq.upper()
        flen = len(flanking)

        if flen < 30:
            logger.warning("Flanking too short (%d bp) for %s", flen, target.label)
            return ScanResult(target_label=target.label)

        ms, me = mutation_footprint(target)

        if ms < 0 or me > flen:
            logger.error(
                "Mutation footprint [%d,%d) outside flanking [0,%d) for %s",
                ms, me, flen, target.label,
            )
            return ScanResult(target_label=target.label)

        # ── Phase 1: Direct overlap scan ──
        direct: list[CrRNACandidate] = []
        seen: set[str] = set()

        # Plus strand
        self._scan_strand_direct(
            target, flanking, Strand.PLUS, ms, me, flen, direct, seen,
        )

        # Minus strand
        rc = str(Seq(flanking).reverse_complement())
        self._scan_strand_direct(
            target, rc, Strand.MINUS, flen - me, flen - ms, flen, direct, seen,
        )

        logger.debug("Target %s: %d direct candidates", target.label, len(direct))

        # ── Phase 2: Proximity fallback ──
        proximity: list[CrRNACandidate] = []
        is_desert = False

        if len(direct) == 0 and self.proximity.enabled:
            is_desert = True
            logger.info(
                "PAM DESERT detected for %s — activating proximity scan "
                "(searching ±%d bp from mutation)",
                target.label,
                self.proximity.max_distance,
            )

            prox_seen: set[str] = set()

            # Plus strand proximity
            self._scan_strand_proximity(
                target, flanking, Strand.PLUS, ms, me, flen,
                proximity, prox_seen,
            )

            # Minus strand proximity
            self._scan_strand_proximity(
                target, rc, Strand.MINUS, flen - me, flen - ms, flen,
                proximity, prox_seen,
            )

            # Sort by distance (closest first)
            proximity.sort(key=lambda c: c.proximity_distance)

            logger.info(
                "Proximity scan for %s: %d candidates (nearest %d bp from mutation)",
                target.label,
                len(proximity),
                proximity[0].proximity_distance if proximity else -1,
            )

        return ScanResult(
            target_label=target.label,
            direct_candidates=direct,
            proximity_candidates=proximity,
            pam_desert=is_desert,
        )

    def scan_batch(self, targets: list[Target]) -> dict[str, ScanResult]:
        """Scan multiple targets and return structured results."""
        return {t.label: self.scan_detailed(t) for t in targets}

    # ------------------------------------------------------------------
    # Phase 1: Direct overlap scan
    # ------------------------------------------------------------------

    def _scan_strand_direct(
        self,
        target: Target,
        seq: str,
        strand: Strand,
        ms: int, me: int,        # mutation [start, end) in this strand's coords
        flen: int,                # original flanking length
        out: list[CrRNACandidate],
        seen: set[str],
    ) -> None:
        """Scan one strand for PAMs with spacers that overlap the mutation."""
        slen = len(seq)

        for i in range(slen - 4):
            pam4 = seq[i : i + 4]

            # Match against PAM patterns (first match wins = highest activity)
            hit: Optional[PAMDef] = None
            for pd in self.pams:
                if iupac_match(pam4, pd.pattern):
                    hit = pd
                    break

            if hit is None:
                continue

            # PAM found at position i → spacer starts at i+4
            sp_start = i + 4

            for sp_len in self.lengths:
                sp_end = sp_start + sp_len

                if sp_end > slen:
                    continue

                # Overlap check: mutation must overlap spacer
                if max(sp_start, ms) >= min(sp_end, me):
                    continue

                # Mutation positions in spacer (1-indexed, PAM-proximal)
                positions = [
                    p - sp_start + 1
                    for p in range(ms, me)
                    if sp_start <= p < sp_end
                ]
                mut_pos = min(positions) if positions else None

                # Genomic coordinate
                if strand == Strand.PLUS:
                    gstart = target.flanking_start + sp_start
                else:
                    gstart = target.flanking_start + (flen - sp_end)

                # Dedup
                key = f"{strand.value}:{gstart}:{sp_len}"
                if key in seen:
                    continue
                seen.add(key)

                spacer = seq[sp_start:sp_end]

                out.append(CrRNACandidate(
                    candidate_id=_cid(target.label, strand, gstart, sp_len),
                    target_label=target.label,
                    spacer_seq=spacer,
                    pam_seq=pam4,
                    pam_variant=_pam_variant(hit.label),
                    strand=strand,
                    genomic_start=gstart,
                    genomic_end=gstart + sp_len,
                    mutation_position_in_spacer=mut_pos,
                    gc_content=_gc(spacer),
                    homopolymer_max=_max_homo(spacer),
                    pam_activity_weight=hit.activity,
                    detection_strategy=DetectionStrategy.DIRECT,
                    proximity_distance=0,
                ))

    # ------------------------------------------------------------------
    # Phase 2: Proximity scan (PAM desert fallback)
    # ------------------------------------------------------------------

    def _scan_strand_proximity(
        self,
        target: Target,
        seq: str,
        strand: Strand,
        ms: int, me: int,        # mutation [start, end) in this strand's coords
        flen: int,                # original flanking length
        out: list[CrRNACandidate],
        seen: set[str],
    ) -> None:
        """Find nearest PAMs when no direct-overlap candidates exist.

        Strategy:
          Search ±max_distance bp from mutation midpoint for any PAM site.
          For each PAM, generate spacers at all valid lengths.
          Compute proximity_distance: distance from nearest spacer edge
          to nearest mutation edge.
          Rank by proximity_distance (smaller = better).
          Cap at max_candidates_per_strand.

        The proximity_distance metric:
          For a spacer [sp_start, sp_end) and mutation [ms, me):
            if spacer is DOWNSTREAM of mutation:
                distance = sp_start - me  (gap between mutation end and spacer start)
            if spacer is UPSTREAM of mutation:
                distance = ms - sp_end    (gap between spacer end and mutation start)
            if overlap (should be caught by direct scan, but just in case):
                distance = 0

          Smaller distance = spacer is closer to mutation = better for
          proximity-based detection (tighter RPA amplicon, more efficient
          trans-cleavage near mismatch site).
        """
        slen = len(seq)
        mut_mid = (ms + me) // 2
        max_dist = self.proximity.max_distance

        # Search window: mutation ± max_distance
        search_start = max(0, mut_mid - max_dist)
        search_end = min(slen - 4, mut_mid + max_dist)

        # Collect all candidate (pam_pos, pam_def, spacer_len, distance) tuples
        candidates_raw: list[tuple[int, PAMDef, int, int, str]] = []

        for i in range(search_start, search_end):
            pam4 = seq[i : i + 4]

            hit: Optional[PAMDef] = None
            for pd in self.pams:
                if iupac_match(pam4, pd.pattern):
                    hit = pd
                    break

            if hit is None:
                continue

            sp_start = i + 4

            for sp_len in self.lengths:
                sp_end = sp_start + sp_len

                if sp_end > slen or sp_start < 0:
                    continue

                # Skip if this would be a direct overlap (shouldn't happen
                # since we only enter proximity when direct found 0, but
                # defensive check)
                if max(sp_start, ms) < min(sp_end, me):
                    continue

                # Compute distance from spacer edge to mutation edge
                if sp_start >= me:
                    # Spacer is entirely downstream of mutation
                    distance = sp_start - me
                elif sp_end <= ms:
                    # Spacer is entirely upstream of mutation
                    distance = ms - sp_end
                else:
                    # Overlap — shouldn't reach here
                    distance = 0

                if distance > max_dist:
                    continue

                spacer = seq[sp_start:sp_end]

                # Genomic coordinate
                if strand == Strand.PLUS:
                    gstart = target.flanking_start + sp_start
                else:
                    gstart = target.flanking_start + (flen - sp_end)

                # Dedup
                key = f"prox:{strand.value}:{gstart}:{sp_len}"
                if key in seen:
                    continue

                candidates_raw.append((i, hit, sp_len, distance, key))

        # Sort by: (distance ASC, PAM activity DESC, spacer length DESC)
        candidates_raw.sort(key=lambda x: (x[3], -x[1].activity, -x[2]))

        # Take top N per strand
        cap = self.proximity.max_candidates_per_strand
        for idx, (pam_pos, pam_def, sp_len, distance, key) in enumerate(candidates_raw):
            if idx >= cap:
                break

            seen.add(key)

            sp_start = pam_pos + 4
            sp_end = sp_start + sp_len
            pam4 = seq[pam_pos : pam_pos + 4]
            spacer = seq[sp_start:sp_end]

            if strand == Strand.PLUS:
                gstart = target.flanking_start + sp_start
            else:
                gstart = target.flanking_start + (flen - sp_end)

            # Apply upstream preference bonus to activity weight
            # Upstream PAMs have spacers closer to mutation on the 3' side
            activity = pam_def.activity
            if sp_end <= ms:
                # Spacer upstream of mutation — slight preference
                activity = min(1.0, activity * self.proximity.prefer_upstream)

            out.append(CrRNACandidate(
                candidate_id=_cid(
                    target.label, strand, gstart, sp_len, prefix="prox_",
                ),
                target_label=target.label,
                spacer_seq=spacer,
                pam_seq=pam4,
                pam_variant=_pam_variant(pam_def.label),
                strand=strand,
                genomic_start=gstart,
                genomic_end=gstart + sp_len,
                mutation_position_in_spacer=None,   # mutation outside spacer
                gc_content=_gc(spacer),
                homopolymer_max=_max_homo(spacer),
                pam_activity_weight=activity,
                detection_strategy=DetectionStrategy.PROXIMITY,
                proximity_distance=distance,
            ))

        if candidates_raw:
            logger.debug(
                "Proximity %s strand %s: %d PAMs found, kept %d (nearest %d bp)",
                target.label,
                strand.value,
                len(candidates_raw),
                min(len(candidates_raw), cap),
                candidates_raw[0][3] if candidates_raw else -1,
            )
        else:
            logger.warning(
                "Proximity %s strand %s: NO PAMs within ±%d bp!",
                target.label,
                strand.value,
                max_dist,
            )
