"""Resolve WHO mutations to genomic coordinates on H37Rv.

Handles the key challenges:
- Gene coordinate lookup from GFF3 annotation
- Codon numbering offset (E. coli legacy numbering → M.tb H37Rv)
- Flanking sequence extraction for downstream PAM scanning
- Reverse complement for genes on minus strand
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from Bio import SeqIO
from Bio.Seq import Seq

from saber.core.constants import FLANKING_WINDOW, H37RV_ACCESSION
from saber.core.types import Mutation, Strand, Target

logger = logging.getLogger(__name__)


class GeneRecord:
    """Coordinates and metadata for a single gene."""

    def __init__(
        self,
        name: str,
        start: int,
        end: int,
        strand: Strand,
        locus_tag: Optional[str] = None,
        codon_offset: int = 0,
    ) -> None:
        self.name = name
        self.start = start          # 0-based, inclusive
        self.end = end              # 0-based, exclusive
        self.strand = strand
        self.locus_tag = locus_tag
        self.codon_offset = codon_offset    # for E.coli→M.tb numbering correction

    def codon_to_genomic(self, codon_number: int) -> int:
        """Convert gene-relative codon number to genomic nucleotide position.

        Returns the position of the first nucleotide of the codon on the genome.
        Applies the codon offset correction.
        """
        adjusted = codon_number - self.codon_offset
        if self.strand == Strand.PLUS:
            return self.start + (adjusted - 1) * 3
        else:
            # Minus strand: codons count from the end
            return self.end - adjusted * 3


class TargetResolver:
    """Resolve mutations to genomic targets.

    Usage:
        resolver = TargetResolver(
            fasta="data/references/H37Rv.fasta",
            gff="data/references/H37Rv.gff3",
        )
        targets = resolver.resolve_all(mutations)
    """

    # Known codon offsets for key TB resistance genes
    # (E. coli numbering convention vs M.tb H37Rv true start)
    CODON_OFFSETS: dict[str, int] = {
        "rpoB": 0,
        "katG": 0,
        "inhA": 0,
        "embB": 0,
        "gyrA": 0,
        "gyrB": 0,
        "rpsL": 0,
        "rrs": 0,
        "pncA": 0,
        "ethA": 0,
    }

    def __init__(
        self,
        fasta: str | Path,
        gff: Optional[str | Path] = None,
        gene_table: Optional[dict[str, GeneRecord]] = None,
    ) -> None:
        self.fasta = Path(fasta)
        self.genome_seq = self._load_genome()
        self.gene_table: dict[str, GeneRecord] = gene_table or {}
        if gff is not None:
            self._parse_gff(Path(gff))

    def resolve(self, mutation: Mutation) -> Optional[Target]:
        """Resolve a single mutation to a genomic Target."""
        gene = self.gene_table.get(mutation.gene)
        if gene is None:
            logger.warning("Gene %s not found in annotation", mutation.gene)
            return None

        genomic_pos = gene.codon_to_genomic(mutation.position)

        # Validate: extract codon at position and check it matches ref
        ref_codon = self._extract_codon(genomic_pos, gene.strand)
        expected_aa = str(Seq(ref_codon).translate())
        if expected_aa != mutation.ref_aa:
            logger.warning(
                "Codon mismatch for %s: found %s (aa=%s), expected aa=%s. "
                "Check codon offset.",
                mutation.label, ref_codon, expected_aa, mutation.ref_aa,
            )

        # Generate alt codon (take most common codon for the alt amino acid)
        alt_codon = self._infer_alt_codon(ref_codon, mutation.alt_aa, gene.strand)

        # Extract flanking sequence
        flank_start = max(0, genomic_pos - FLANKING_WINDOW)
        flank_end = min(len(self.genome_seq), genomic_pos + 3 + FLANKING_WINDOW)
        flanking = str(self.genome_seq[flank_start:flank_end])

        return Target(
            mutation=mutation,
            chrom=H37RV_ACCESSION,
            genomic_pos=genomic_pos,
            ref_codon=ref_codon,
            alt_codon=alt_codon,
            flanking_seq=flanking,
            flanking_start=flank_start,
        )

    def resolve_all(self, mutations: list[Mutation]) -> list[Target]:
        """Resolve a batch of mutations. Skips unresolvable entries."""
        targets = []
        for mut in mutations:
            t = self.resolve(mut)
            if t is not None:
                targets.append(t)
        logger.info("Resolved %d / %d mutations", len(targets), len(mutations))
        return targets

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_genome(self) -> Seq:
        record = SeqIO.read(self.fasta, "fasta")
        logger.info("Loaded genome: %s (%d bp)", record.id, len(record.seq))
        return record.seq

    def _parse_gff(self, gff_path: Path) -> None:
        """Minimal GFF3 parser for gene coordinates."""
        with open(gff_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 9 or parts[2] != "gene":
                    continue
                attrs = dict(
                    kv.split("=", 1)
                    for kv in parts[8].split(";")
                    if "=" in kv
                )
                name = attrs.get("Name", attrs.get("gene", ""))
                locus_tag = attrs.get("locus_tag", "")
                if not name:
                    continue
                strand = Strand.PLUS if parts[6] == "+" else Strand.MINUS
                offset = self.CODON_OFFSETS.get(name, 0)
                self.gene_table[name] = GeneRecord(
                    name=name,
                    start=int(parts[3]) - 1,        # GFF is 1-based
                    end=int(parts[4]),                # GFF end is inclusive, we use exclusive
                    strand=strand,
                    locus_tag=locus_tag,
                    codon_offset=offset,
                )
        logger.info("Loaded %d genes from GFF", len(self.gene_table))

    def _extract_codon(self, genomic_pos: int, strand: Strand) -> str:
        codon = str(self.genome_seq[genomic_pos : genomic_pos + 3])
        if strand == Strand.MINUS:
            codon = str(Seq(codon).reverse_complement())
        return codon.upper()

    @staticmethod
    def _infer_alt_codon(ref_codon: str, alt_aa: str, strand: Strand) -> str:
        """Find the single-nucleotide change that converts ref_codon to code for alt_aa.

        Tries all 9 possible single-nt substitutions. If multiple work,
        picks the one matching a transition (more common than transversion).
        """
        transitions = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
        candidates: list[tuple[str, bool]] = []

        for i in range(3):
            for nt in "ATGC":
                if nt == ref_codon[i]:
                    continue
                alt = ref_codon[:i] + nt + ref_codon[i + 1 :]
                aa = str(Seq(alt).translate())
                if aa == alt_aa:
                    is_transition = (ref_codon[i], nt) in transitions
                    candidates.append((alt, is_transition))

        if not candidates:
            logger.warning("No single-nt change produces %s from %s", alt_aa, ref_codon)
            return ref_codon  # fallback

        # Prefer transitions
        candidates.sort(key=lambda x: not x[1])
        return candidates[0][0]
