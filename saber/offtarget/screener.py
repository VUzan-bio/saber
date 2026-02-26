"""Off-target screening for crRNA candidates.

Aligns each spacer against:
1. The full H37Rv genome (self-targeting check)
2. A representative human genome (specificity for blood-based detection)

Uses Bowtie2 in end-to-end mode with configurable mismatch tolerance.
For multiplex panels, also checks cross-reactivity between crRNAs
and non-target amplicon sequences.

Requires pre-built Bowtie2 indices:
    bowtie2-build H37Rv.fasta H37Rv
    bowtie2-build GRCh38.fasta GRCh38
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from saber.core.constants import BOWTIE2_MAX_MISMATCHES, BOWTIE2_SEED_LENGTH
from saber.core.types import CrRNACandidate, OffTargetHit, OffTargetReport

logger = logging.getLogger(__name__)


class OffTargetScreener:
    """Screen crRNA spacers for off-target hits.

    Usage:
        screener = OffTargetScreener(
            mtb_index="data/references/H37Rv",
            human_index="data/references/GRCh38",
        )
        report = screener.screen(candidate)
        reports = screener.screen_batch(candidates)
    """

    def __init__(
        self,
        mtb_index: str | Path,
        human_index: Optional[str | Path] = None,
        max_mismatches: int = BOWTIE2_MAX_MISMATCHES,
        threads: int = 4,
    ) -> None:
        self.mtb_index = str(mtb_index)
        self.human_index = str(human_index) if human_index else None
        self.max_mismatches = max_mismatches
        self.threads = threads

    def screen(self, candidate: CrRNACandidate) -> OffTargetReport:
        """Screen a single candidate against all reference genomes."""
        mtb_hits = self._align(candidate.spacer_seq, self.mtb_index)

        # Remove the on-target hit (exact match at the designed position)
        mtb_hits = [
            h for h in mtb_hits
            if not (h.mismatches == 0 and h.hit_start == candidate.genomic_start)
        ]

        human_hits: list[OffTargetHit] = []
        if self.human_index:
            human_hits = self._align(candidate.spacer_seq, self.human_index)

        risky = [h for h in mtb_hits + human_hits if h.mismatches <= self.max_mismatches]

        return OffTargetReport(
            candidate_id=candidate.candidate_id,
            mtb_hits=[self._to_hit(candidate.candidate_id, h) for h in mtb_hits],
            human_hits=[self._to_hit(candidate.candidate_id, h) for h in human_hits],
            is_clean=len(risky) == 0,
        )

    def screen_batch(self, candidates: list[CrRNACandidate]) -> list[OffTargetReport]:
        """Screen multiple candidates. Uses batch alignment for efficiency."""
        if not candidates:
            return []

        # Batch alignment: write all spacers to one FASTA, align once
        reports: dict[str, OffTargetReport] = {}

        mtb_results = self._align_batch(
            {c.candidate_id: c.spacer_seq for c in candidates},
            self.mtb_index,
        )

        human_results: dict[str, list[_RawHit]] = {}
        if self.human_index:
            human_results = self._align_batch(
                {c.candidate_id: c.spacer_seq for c in candidates},
                self.human_index,
            )

        for c in candidates:
            cid = c.candidate_id
            mtb_hits = mtb_results.get(cid, [])
            # Filter on-target
            mtb_hits = [
                h for h in mtb_hits
                if not (h.mismatches == 0 and h.start == c.genomic_start)
            ]
            human_hits = human_results.get(cid, [])

            all_hits = mtb_hits + human_hits
            risky = [h for h in all_hits if h.mismatches <= self.max_mismatches]

            reports[cid] = OffTargetReport(
                candidate_id=cid,
                mtb_hits=[self._to_hit(cid, h) for h in mtb_hits],
                human_hits=[self._to_hit(cid, h) for h in human_hits],
                is_clean=len(risky) == 0,
            )

        return [reports[c.candidate_id] for c in candidates]

    def check_cross_reactivity(
        self,
        spacers: dict[str, str],
        amplicons: dict[str, str],
    ) -> dict[str, list[OffTargetHit]]:
        """Check if any crRNA spacer aligns to a non-target amplicon.

        For multiplex panels: each crRNA should only activate on its own amplicon.
        """
        cross_hits: dict[str, list[OffTargetHit]] = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            # Build temporary index from amplicon sequences
            amplicon_fasta = Path(tmpdir) / "amplicons.fasta"
            with open(amplicon_fasta, "w") as f:
                for name, seq in amplicons.items():
                    f.write(f">{name}\n{seq}\n")

            idx_prefix = str(Path(tmpdir) / "amplicons")
            subprocess.run(
                ["bowtie2-build", "--quiet", str(amplicon_fasta), idx_prefix],
                check=True, capture_output=True,
            )

            for cid, spacer in spacers.items():
                hits = self._align(spacer, idx_prefix)
                # Flag hits to non-self amplicons
                cross = [h for h in hits if h.chrom != cid and h.mismatches <= self.max_mismatches]
                if cross:
                    cross_hits[cid] = [self._to_hit(cid, h) for h in cross]

        return cross_hits

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    class _RawHit:
        """Minimal parsed alignment hit."""
        __slots__ = ("chrom", "start", "end", "mismatches", "score")

        def __init__(self, chrom: str, start: int, end: int, mismatches: int, score: float):
            self.chrom = chrom
            self.start = start
            self.end = end
            self.mismatches = mismatches
            self.score = score

    def _align(self, spacer: str, index: str) -> list[_RawHit]:
        """Align a single spacer using Bowtie2."""
        try:
            result = subprocess.run(
                [
                    "bowtie2",
                    "-x", index,
                    "-c", spacer,
                    "--end-to-end",
                    f"-L {BOWTIE2_SEED_LENGTH}",
                    "-k", "10",                     # report up to 10 alignments
                    "--no-hd",                       # suppress header
                    f"-p {self.threads}",
                    "--quiet",
                ],
                capture_output=True, text=True, timeout=30,
            )
            return self._parse_sam(result.stdout)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning("Bowtie2 alignment failed: %s", e)
            return []

    def _align_batch(
        self,
        spacers: dict[str, str],
        index: str,
    ) -> dict[str, list[_RawHit]]:
        """Batch-align multiple spacers."""
        results: dict[str, list[_RawHit]] = {cid: [] for cid in spacers}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            for cid, seq in spacers.items():
                f.write(f">{cid}\n{seq}\n")
            fasta_path = f.name

        try:
            result = subprocess.run(
                [
                    "bowtie2",
                    "-x", index,
                    "-f", fasta_path,
                    "--end-to-end",
                    f"-L {BOWTIE2_SEED_LENGTH}",
                    "-k", "10",
                    "--no-hd",
                    f"-p {self.threads}",
                    "--quiet",
                ],
                capture_output=True, text=True, timeout=120,
            )
            for line in result.stdout.strip().split("\n"):
                if not line or line.startswith("@"):
                    continue
                fields = line.split("\t")
                if len(fields) < 10:
                    continue
                cid = fields[0]
                hit = self._parse_sam_line(fields)
                if hit and cid in results:
                    results[cid].append(hit)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning("Batch Bowtie2 alignment failed: %s", e)

        return results

    def _parse_sam(self, sam_output: str) -> list[_RawHit]:
        hits: list[OffTargetScreener._RawHit] = []
        for line in sam_output.strip().split("\n"):
            if not line or line.startswith("@"):
                continue
            fields = line.split("\t")
            if len(fields) < 10:
                continue
            hit = self._parse_sam_line(fields)
            if hit:
                hits.append(hit)
        return hits

    @staticmethod
    def _parse_sam_line(fields: list[str]) -> Optional[_RawHit]:
        """Parse a single SAM alignment line."""
        flag = int(fields[1])
        if flag & 4:  # unmapped
            return None
        chrom = fields[2]
        start = int(fields[3]) - 1  # SAM is 1-based
        seq_len = len(fields[9])
        end = start + seq_len

        # Count mismatches from XM tag or NM tag
        mismatches = 0
        for tag in fields[11:]:
            if tag.startswith("XM:i:") or tag.startswith("NM:i:"):
                mismatches = int(tag.split(":")[-1])
                break

        score = float(fields[4])  # MAPQ

        return OffTargetScreener._RawHit(chrom, start, end, mismatches, score)

    @staticmethod
    def _to_hit(candidate_id: str, raw: _RawHit) -> OffTargetHit:
        return OffTargetHit(
            candidate_id=candidate_id,
            hit_chrom=raw.chrom,
            hit_start=raw.start,
            hit_end=raw.end,
            mismatches=raw.mismatches,
            alignment_score=raw.score,
        )
