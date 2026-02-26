"""Scan target flanking sequences for Cas12a-compatible PAM sites.

For each PAM hit, extract the downstream spacer and check whether the
resistance mutation falls within the seed region (positions 1-8 from PAM).

PAM orientation for Cas12a:
    5'- [TTTV]----[spacer 20-24nt]---- 3'
                   ^pos1            ^pos20+
    The seed region is positions 1-8 (PAM-proximal).

Both strands are scanned. For the minus strand, the flanking sequence is
reverse-complemented before scanning.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

from Bio.Seq import Seq

from saber.core.constants import (
    ASCAS12A_PAM,
    ENASCAS12A_PAMS,
    SEED_REGION_END,
    SPACER_LENGTH_DEFAULT,
    pam_matches,
)
from saber.core.types import CrRNACandidate, PAMVariant, Strand, Target

logger = logging.getLogger(__name__)


def _gc_content(seq: str) -> float:
    s = seq.upper()
    gc = s.count("G") + s.count("C")
    return gc / len(s) if s else 0.0


def _max_homopolymer(seq: str) -> int:
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


def _candidate_id(target_label: str, strand: Strand, genomic_start: int) -> str:
    """Deterministic ID from target + position."""
    raw = f"{target_label}:{strand.value}:{genomic_start}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


class PAMScanner:
    """Scan for Cas12a PAM sites in target flanking sequences.

    Usage:
        scanner = PAMScanner(spacer_length=20, use_enascas12a=True)
        candidates = scanner.scan(target)
    """

    def __init__(
        self,
        spacer_length: int = SPACER_LENGTH_DEFAULT,
        use_enascas12a: bool = True,
    ) -> None:
        self.spacer_length = spacer_length
        self.pam_patterns: list[tuple[str, PAMVariant]] = [(ASCAS12A_PAM, PAMVariant.TTTV)]
        if use_enascas12a:
            for pam in ENASCAS12A_PAMS:
                self.pam_patterns.append((pam, PAMVariant(pam)))

    def scan(self, target: Target) -> list[CrRNACandidate]:
        """Scan both strands of the flanking sequence for PAM sites."""
        candidates: list[CrRNACandidate] = []

        # Mutation position relative to flanking sequence
        mut_offset = target.genomic_pos - target.flanking_start

        # Plus strand
        candidates.extend(
            self._scan_strand(target, target.flanking_seq, Strand.PLUS, mut_offset)
        )

        # Minus strand (reverse complement, adjust offset)
        rc_seq = str(Seq(target.flanking_seq).reverse_complement())
        rc_mut_offset = len(target.flanking_seq) - mut_offset - 3
        candidates.extend(
            self._scan_strand(target, rc_seq, Strand.MINUS, rc_mut_offset)
        )

        logger.debug(
            "Target %s: found %d PAM hits across both strands",
            target.label, len(candidates),
        )
        return candidates

    def scan_batch(self, targets: list[Target]) -> dict[str, list[CrRNACandidate]]:
        """Scan multiple targets. Returns {target_label: [candidates]}."""
        return {t.label: self.scan(t) for t in targets}

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _scan_strand(
        self,
        target: Target,
        seq: str,
        strand: Strand,
        mut_offset: int,
    ) -> list[CrRNACandidate]:
        candidates: list[CrRNACandidate] = []
        pam_len = 4
        spacer_end = pam_len + self.spacer_length

        for i in range(len(seq) - spacer_end):
            pam_seq = seq[i : i + pam_len]

            for pattern, variant in self.pam_patterns:
                if not pam_matches(pam_seq, pattern):
                    continue

                spacer = seq[i + pam_len : i + spacer_end]

                # Where does the mutation fall in this spacer?
                # Position 1 = first nt after PAM (PAM-proximal)
                mut_pos_in_spacer = self._mutation_position(
                    pam_start=i, pam_len=pam_len,
                    spacer_len=self.spacer_length, mut_offset=mut_offset,
                )

                if mut_pos_in_spacer is None:
                    continue  # mutation not in this spacer

                # Convert local position to genomic
                if strand == Strand.PLUS:
                    genomic_start = target.flanking_start + i + pam_len
                else:
                    genomic_start = (
                        target.flanking_start
                        + len(seq) - (i + spacer_end)
                    )

                cid = _candidate_id(target.label, strand, genomic_start)

                candidates.append(CrRNACandidate(
                    candidate_id=cid,
                    target_label=target.label,
                    spacer_seq=spacer.upper(),
                    pam_seq=pam_seq.upper(),
                    pam_variant=variant,
                    strand=strand,
                    genomic_start=genomic_start,
                    genomic_end=genomic_start + self.spacer_length,
                    mutation_position_in_spacer=mut_pos_in_spacer,
                    gc_content=_gc_content(spacer),
                    homopolymer_max=_max_homopolymer(spacer),
                ))
                break  # don't double-count same PAM under multiple patterns

        return candidates

    @staticmethod
    def _mutation_position(
        pam_start: int,
        pam_len: int,
        spacer_len: int,
        mut_offset: int,
    ) -> Optional[int]:
        """Compute 1-indexed mutation position in spacer (from PAM-proximal end).

        Returns None if mutation is not within the spacer.
        """
        spacer_start = pam_start + pam_len
        spacer_end = spacer_start + spacer_len

        # Mutation covers 3 nt (codon). Check if any nt overlaps the spacer.
        for nt in range(3):
            pos = mut_offset + nt
            if spacer_start <= pos < spacer_end:
                return pos - spacer_start + 1  # 1-indexed from PAM-proximal

        return None
