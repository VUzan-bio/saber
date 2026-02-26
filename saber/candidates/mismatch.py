"""Generate WT/MUT spacer pairs for discrimination analysis.

For each crRNA candidate, produces the corresponding wild-type spacer and
every clinically relevant mismatch variant. These pairs are the fundamental
unit for predicting (and later measuring) single-nucleotide discrimination.

A candidate designed to detect a mutant will have:
- MUT spacer: perfect match to mutant target → high activity (desired)
- WT spacer: single mismatch to wild-type target → low activity (desired)

The discrimination ratio = MUT_activity / WT_activity should be >> 1.
"""

from __future__ import annotations

import logging

from saber.core.types import CrRNACandidate, MismatchPair, Target

logger = logging.getLogger(__name__)

_COMPLEMENTS = {"A": "T", "T": "A", "G": "C", "C": "G"}
_BASES = set("ATGC")


class MismatchGenerator:
    """Generate mismatch pairs for discrimination analysis.

    Usage:
        gen = MismatchGenerator()
        pairs = gen.generate(candidate, target)
        all_pairs = gen.generate_exhaustive(candidate)
    """

    def generate(
        self,
        candidate: CrRNACandidate,
        target: Target,
    ) -> MismatchPair:
        """Generate the primary WT/MUT pair for a candidate.

        The candidate spacer matches the mutant sequence.
        The WT spacer has the reference nucleotide at the mutation position.
        """
        pos_idx = candidate.mutation_position_in_spacer - 1  # 0-indexed

        # The candidate spacer is designed against the mutant
        mut_spacer = candidate.spacer_seq

        # Determine what the WT nucleotide would be at this position
        mut_nt = mut_spacer[pos_idx]

        # From the target's ref/alt codons, find the WT nucleotide
        # The mutation position in spacer corresponds to a specific codon position
        wt_nt = self._resolve_wt_nucleotide(target, candidate)

        if wt_nt == mut_nt:
            logger.warning(
                "Candidate %s: WT and MUT nucleotides are identical at position %d",
                candidate.candidate_id, candidate.mutation_position_in_spacer,
            )

        wt_spacer = mut_spacer[:pos_idx] + wt_nt + mut_spacer[pos_idx + 1:]
        mismatch_type = f"{wt_nt}>{mut_nt}"

        return MismatchPair(
            candidate_id=candidate.candidate_id,
            wt_spacer=wt_spacer,
            mut_spacer=mut_spacer,
            mismatch_position=candidate.mutation_position_in_spacer,
            mismatch_type=mismatch_type,
        )

    def generate_exhaustive(
        self,
        candidate: CrRNACandidate,
    ) -> list[MismatchPair]:
        """Generate all possible single-mismatch variants at every position.

        Produces 3 × spacer_length pairs (3 possible substitutions per position).
        Used for computing the full discrimination landscape / heatmap.
        """
        pairs: list[MismatchPair] = []
        spacer = candidate.spacer_seq

        for pos in range(len(spacer)):
            ref_nt = spacer[pos]
            for alt_nt in _BASES - {ref_nt}:
                mismatched = spacer[:pos] + alt_nt + spacer[pos + 1:]
                pairs.append(MismatchPair(
                    candidate_id=candidate.candidate_id,
                    wt_spacer=mismatched,  # "WT" = the mismatched version
                    mut_spacer=spacer,      # "MUT" = perfect match (the design)
                    mismatch_position=pos + 1,
                    mismatch_type=f"{ref_nt}>{alt_nt}",
                ))

        return pairs

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_wt_nucleotide(target: Target, candidate: CrRNACandidate) -> str:
        """Determine the wild-type nucleotide at the mutation position in the spacer.

        Compares ref_codon and alt_codon from the Target to find which
        nucleotide differs, then maps it to the spacer coordinate system.
        """
        ref = target.ref_codon
        alt = target.alt_codon

        # Find the differing position within the codon
        for i in range(3):
            if ref[i] != alt[i]:
                return ref[i]

        # Fallback: if codons are identical (shouldn't happen), return first nt
        logger.warning("ref_codon == alt_codon for target %s", target.label)
        return ref[0]
