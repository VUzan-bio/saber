"""RPA primer design for CRISPR-Cas12a diagnostic amplicons.

RPA primers differ from PCR primers:
- Longer: 30-35 nt (recombinase binding requirement)
- Lower Tm range: 60-65°C
- Short amplicons: 100-200 bp (isothermal amplification efficiency)
- Sensitive to primer dimers in multiplex (all 2N primers in one pot)

For the 14-plex panel, there are 28 primers with C(28,2) = 378 pairwise
interactions to check for dimer formation.

Integration with Module 5: Primer compatibility is a hard constraint on
multiplex selection. A crRNA candidate with perfect efficiency but
incompatible primers is unusable.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp as mt

from saber.core.constants import (
    PRIMER_DIMER_DG_THRESHOLD,
    RPA_AMPLICON_MAX,
    RPA_AMPLICON_MIN,
    RPA_PRIMER_LENGTH_MAX,
    RPA_PRIMER_LENGTH_MIN,
    RPA_TM_MAX,
    RPA_TM_MIN,
)
from saber.core.types import CrRNACandidate, RPAPrimer, RPAPrimerPair

logger = logging.getLogger(__name__)


class RPAPrimerDesigner:
    """Design RPA primers flanking crRNA target sites.

    Usage:
        designer = RPAPrimerDesigner()
        pair = designer.design(candidate, genome_seq)
        compatible = designer.check_multiplex_dimers(all_pairs)
    """

    def __init__(
        self,
        primer_len_min: int = RPA_PRIMER_LENGTH_MIN,
        primer_len_max: int = RPA_PRIMER_LENGTH_MAX,
        tm_min: float = RPA_TM_MIN,
        tm_max: float = RPA_TM_MAX,
        amplicon_min: int = RPA_AMPLICON_MIN,
        amplicon_max: int = RPA_AMPLICON_MAX,
        dimer_dg_threshold: float = PRIMER_DIMER_DG_THRESHOLD,
    ) -> None:
        self.primer_len_min = primer_len_min
        self.primer_len_max = primer_len_max
        self.tm_min = tm_min
        self.tm_max = tm_max
        self.amplicon_min = amplicon_min
        self.amplicon_max = amplicon_max
        self.dimer_dg_threshold = dimer_dg_threshold

    def design(
        self,
        candidate: CrRNACandidate,
        genome_seq: str,
    ) -> list[RPAPrimerPair]:
        """Design RPA primer pairs flanking a crRNA target site.

        Returns ranked list of valid primer pairs.
        """
        target_start = candidate.genomic_start
        target_end = candidate.genomic_end

        fwd_candidates = self._scan_forward_primers(genome_seq, target_start)
        rev_candidates = self._scan_reverse_primers(genome_seq, target_end)

        pairs: list[RPAPrimerPair] = []

        for fwd in fwd_candidates:
            for rev in rev_candidates:
                amplicon_len = rev.amplicon_end - fwd.amplicon_start
                if not (self.amplicon_min <= amplicon_len <= self.amplicon_max):
                    continue

                # Check self-dimer potential
                dimer_dg = self._check_dimer(fwd.seq, rev.seq)

                pairs.append(RPAPrimerPair(
                    fwd=fwd,
                    rev=rev,
                    dimer_dg=dimer_dg,
                ))

        # Filter and rank
        pairs = [p for p in pairs if self._is_valid_pair(p)]
        pairs.sort(key=self._pair_score, reverse=True)

        logger.debug(
            "Candidate %s: %d valid primer pairs",
            candidate.candidate_id, len(pairs),
        )
        return pairs

    def check_multiplex_dimers(
        self,
        primer_pairs: list[RPAPrimerPair],
    ) -> list[list[Optional[float]]]:
        """Check all pairwise primer-dimer interactions in a multiplex panel.

        Returns an NxN matrix where N = 2 * len(primer_pairs) (fwd + rev for each).
        Matrix[i][j] = dG of the most stable dimer between primer i and primer j.
        None if dG > threshold (acceptable).
        """
        all_primers = []
        for pp in primer_pairs:
            all_primers.append(pp.fwd.seq)
            all_primers.append(pp.rev.seq)

        n = len(all_primers)
        matrix: list[list[Optional[float]]] = [[None] * n for _ in range(n)]

        conflicts = 0
        for i in range(n):
            for j in range(i + 1, n):
                dg = self._check_dimer(all_primers[i], all_primers[j])
                if dg is not None and dg < self.dimer_dg_threshold:
                    matrix[i][j] = dg
                    matrix[j][i] = dg
                    conflicts += 1

        if conflicts > 0:
            logger.warning(
                "Multiplex dimer check: %d conflicts among %d primers",
                conflicts, n,
            )
        return matrix

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _scan_forward_primers(
        self, genome: str, target_start: int,
    ) -> list[RPAPrimer]:
        """Scan upstream of target for forward primers."""
        primers: list[RPAPrimer] = []
        # Search window: amplicon_max upstream of target start
        window_start = max(0, target_start - self.amplicon_max)
        window_end = target_start - 10  # leave gap before target

        for start in range(window_start, window_end):
            for length in range(self.primer_len_min, self.primer_len_max + 1):
                end = start + length
                if end > len(genome):
                    break
                seq = genome[start:end].upper()
                if not set(seq).issubset({"A", "T", "G", "C"}):
                    continue

                tm_val = self._compute_tm(seq)
                if not (self.tm_min <= tm_val <= self.tm_max):
                    continue

                primers.append(RPAPrimer(
                    seq=seq,
                    tm=tm_val,
                    direction="fwd",
                    amplicon_start=start,
                    amplicon_end=start + self.amplicon_max,  # placeholder
                ))

        return primers[:50]  # cap to avoid combinatorial explosion

    def _scan_reverse_primers(
        self, genome: str, target_end: int,
    ) -> list[RPAPrimer]:
        """Scan downstream of target for reverse primers."""
        primers: list[RPAPrimer] = []
        window_start = target_end + 10
        window_end = min(len(genome), target_end + self.amplicon_max)

        for end in range(window_start + self.primer_len_min, window_end):
            for length in range(self.primer_len_min, self.primer_len_max + 1):
                start = end - length
                if start < 0:
                    continue
                seq = str(Seq(genome[start:end].upper()).reverse_complement())
                if not set(seq).issubset({"A", "T", "G", "C"}):
                    continue

                tm_val = self._compute_tm(seq)
                if not (self.tm_min <= tm_val <= self.tm_max):
                    continue

                primers.append(RPAPrimer(
                    seq=seq,
                    tm=tm_val,
                    direction="rev",
                    amplicon_start=target_end - self.amplicon_max,  # placeholder
                    amplicon_end=end,
                ))

        return primers[:50]

    @staticmethod
    def _compute_tm(seq: str) -> float:
        """Compute melting temperature using nearest-neighbour method."""
        return mt.Tm_NN(Seq(seq), nn_table=mt.DNA_NN3)

    @staticmethod
    def _check_dimer(seq1: str, seq2: str) -> Optional[float]:
        """Check for primer dimer formation using Primer3 ntthal.

        Returns dG in kcal/mol, or None if ntthal is unavailable.
        """
        try:
            result = subprocess.run(
                ["ntthal", "-a", "END1", "-s1", seq1, "-s2", seq2],
                capture_output=True, text=True, timeout=5,
            )
            # Parse dG from ntthal output
            for line in result.stdout.strip().split("\n"):
                if "dG" in line:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p == "dG" and i + 2 < len(parts):
                            return float(parts[i + 2])
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass
        return None

    def _is_valid_pair(self, pair: RPAPrimerPair) -> bool:
        """Check if a primer pair passes all hard constraints."""
        if pair.dimer_dg is not None and pair.dimer_dg < self.dimer_dg_threshold:
            return False
        return True

    @staticmethod
    def _pair_score(pair: RPAPrimerPair) -> float:
        """Score a primer pair. Higher = better."""
        # Prefer Tm close to 62.5°C (midpoint of range)
        tm_opt = 62.5
        fwd_tm_score = 1.0 - abs(pair.fwd.tm - tm_opt) / 5.0
        rev_tm_score = 1.0 - abs(pair.rev.tm - tm_opt) / 5.0
        # Prefer shorter amplicons (more efficient RPA)
        amp_score = 1.0 - (pair.amplicon_length - RPA_AMPLICON_MIN) / (RPA_AMPLICON_MAX - RPA_AMPLICON_MIN)
        return fwd_tm_score + rev_tm_score + amp_score
