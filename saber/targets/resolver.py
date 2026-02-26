"""Resolve clinical mutations to genomic coordinates on any bacterial reference.

Organism-agnostic, mutation-type-complete, numbering-convention-aware.

Supports:
  - Any bacterial reference genome (not just H37Rv)
  - All mutation types: amino acid substitutions, nucleotide SNPs,
    insertions, deletions, promoter mutations, rRNA mutations,
    frameshifts, large deletions, multi-nucleotide variants
  - Multiple annotation formats: GFF3, GenBank
  - Auto-detection of codon numbering offsets (E. coli legacy, etc.)
  - Multi-reference resolution (e.g. H37Rv + clinical isolate)
  - Validation engine: cross-checks every resolved position

Architecture:
  GenomeStore        — loads and indexes one or more reference genomes
  GeneRecord         — coordinates, strand, product for a single gene
  OffsetResolver     — 5-strategy cascade for codon numbering correction
  MutationClassifier — infers mutation type from heterogeneous notation
  TargetResolver     — public API, wires everything together

References:
  - WHO 2023 Catalogue of mutations (mixed numbering conventions)
  - Miotto et al., Genome Medicine 2017 (TB mutation nomenclature)
  - Andre et al., Lancet Microbe 2022 (standardised mutation notation)
  - CRISPResso2 (Clement et al., Nature Biotech 2019) — indel handling
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

from Bio import SeqIO
from Bio.Seq import Seq

from saber.core.constants import FLANKING_WINDOW, H37RV_ACCESSION
from saber.core.types import Mutation, Strand, Target

logger = logging.getLogger(__name__)


# ======================================================================
# Mutation classification
# ======================================================================

class MutationType(str, Enum):
    """Exhaustive classification of clinical mutation types."""
    AA_SUBSTITUTION = "aa_substitution"     # S450L, H445Y
    NUCLEOTIDE_SNP = "nucleotide_snp"       # c.1349C>T, A1401G (rRNA)
    INSERTION = "insertion"                  # c.516_517insG
    DELETION = "deletion"                   # c.171delC
    LARGE_DELETION = "large_deletion"       # full gene deletion, IS6110
    MNV = "mnv"                             # multi-nucleotide variant
    PROMOTER = "promoter"                   # inhA c.-15C>T, fabG1 upstream
    RRNA = "rrna"                           # rrs A1401G, rrl C2270T
    FRAMESHIFT = "frameshift"               # pncA various
    INTERGENIC = "intergenic"               # mabA-inhA intergenic
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ClassifiedMutation:
    """Mutation with inferred type and parsed components."""
    original: Mutation
    mutation_type: MutationType
    # Parsed fields (populated depending on type)
    codon_position: Optional[int] = None       # for AA substitutions
    nucleotide_position: Optional[int] = None  # for nt-level mutations
    ref_base: Optional[str] = None             # for nt SNPs
    alt_base: Optional[str] = None
    inserted_seq: Optional[str] = None         # for insertions
    deleted_length: Optional[int] = None       # for deletions
    is_promoter: bool = False
    promoter_distance: Optional[int] = None    # distance upstream of start


class MutationClassifier:
    """Infer mutation type from heterogeneous clinical notation.

    Handles WHO catalogue format, HGVS-like notation, and common
    shorthand used in TB literature.

    Examples:
        S450L           -> AA_SUBSTITUTION (codon 450)
        c.1349C>T       -> NUCLEOTIDE_SNP
        A1401G          -> RRNA (if gene is rrs/rrl)
        c.-15C>T        -> PROMOTER
        c.516_517insG   -> INSERTION
        c.171delC       -> DELETION
    """

    # rRNA gene names (nucleotide-level mutations, not codon-based)
    RRNA_GENES: set[str] = {"rrs", "rrl", "rrn", "rrf"}

    # Genes with known promoter mutation relevance
    PROMOTER_GENES: set[str] = {"inhA", "fabG1", "eis", "ahpC", "embA", "Rv0678"}

    def classify(self, mutation: Mutation) -> ClassifiedMutation:
        """Classify a single mutation."""
        gene = mutation.gene
        nuc = getattr(mutation, "nucleotide_change", None) or ""
        position = mutation.position
        ref_aa = mutation.ref_aa
        alt_aa = mutation.alt_aa

        # --- rRNA genes: always nucleotide-level ---
        if gene.lower() in {g.lower() for g in self.RRNA_GENES}:
            return self._classify_rrna(mutation, nuc)

        # --- Promoter: negative position or c.-N notation ---
        if position < 0 or self._is_promoter_notation(nuc):
            return self._classify_promoter(mutation, nuc)

        # --- Nucleotide-level annotation ---
        if nuc:
            nuc_lower = nuc.lower()

            # Insertion
            if "ins" in nuc_lower:
                return self._classify_insertion(mutation, nuc)

            # Deletion
            if "del" in nuc_lower:
                del_size = self._parse_deletion_size(nuc)
                if del_size > 50:
                    return ClassifiedMutation(
                        original=mutation,
                        mutation_type=MutationType.LARGE_DELETION,
                        deleted_length=del_size,
                    )
                return self._classify_deletion(mutation, nuc, del_size)

            # Frameshift
            if "fs" in nuc_lower or alt_aa == "*":
                return ClassifiedMutation(
                    original=mutation,
                    mutation_type=MutationType.FRAMESHIFT,
                    codon_position=position,
                )

            # Nucleotide SNP (c.NNN_X>Y)
            if ">" in nuc:
                return self._classify_nt_snp(mutation, nuc)

        # --- Large deletion: gene-level annotation ---
        if alt_aa in ("\u0394", "del", "-", "") and ref_aa in ("", "-", "\u0394"):
            return ClassifiedMutation(
                original=mutation,
                mutation_type=MutationType.LARGE_DELETION,
            )

        # --- Amino acid substitution (default) ---
        if ref_aa and alt_aa and len(ref_aa) == 1 and len(alt_aa) == 1:
            return ClassifiedMutation(
                original=mutation,
                mutation_type=MutationType.AA_SUBSTITUTION,
                codon_position=position,
            )

        # --- Fallback ---
        logger.debug(
            "Could not classify mutation %s, defaulting to UNKNOWN",
            mutation.label,
        )
        return ClassifiedMutation(
            original=mutation,
            mutation_type=MutationType.UNKNOWN,
            codon_position=position,
        )

    def _classify_rrna(self, mutation: Mutation, nuc: str) -> ClassifiedMutation:
        """rRNA mutations use nucleotide position directly."""
        nt_pos = mutation.position
        ref_base, alt_base = None, None

        # rRNA mutations often encoded as A1401G: ref_aa=A, alt_aa=G
        if mutation.ref_aa in "ATGCU" and mutation.alt_aa in "ATGCU":
            ref_base = mutation.ref_aa
            alt_base = mutation.alt_aa
        elif ">" in nuc:
            match = re.search(r"([ATGC])>([ATGC])", nuc, re.IGNORECASE)
            if match:
                ref_base = match.group(1).upper()
                alt_base = match.group(2).upper()

        return ClassifiedMutation(
            original=mutation,
            mutation_type=MutationType.RRNA,
            nucleotide_position=nt_pos,
            ref_base=ref_base,
            alt_base=alt_base,
        )

    def _classify_promoter(self, mutation: Mutation, nuc: str) -> ClassifiedMutation:
        """Promoter mutations: position relative to gene start."""
        distance = abs(mutation.position)
        ref_base, alt_base = None, None

        if ">" in nuc:
            match = re.search(r"([ATGC])>([ATGC])", nuc, re.IGNORECASE)
            if match:
                ref_base = match.group(1).upper()
                alt_base = match.group(2).upper()
        elif mutation.ref_aa in "ATGC" and mutation.alt_aa in "ATGC":
            ref_base = mutation.ref_aa
            alt_base = mutation.alt_aa

        return ClassifiedMutation(
            original=mutation,
            mutation_type=MutationType.PROMOTER,
            nucleotide_position=-distance,
            promoter_distance=distance,
            is_promoter=True,
            ref_base=ref_base,
            alt_base=alt_base,
        )

    def _classify_insertion(self, mutation: Mutation, nuc: str) -> ClassifiedMutation:
        """Parse insertion from c.NNN_NNNinsXXX notation."""
        match = re.search(r"ins([ATGC]+)", nuc, re.IGNORECASE)
        inserted = match.group(1).upper() if match else None

        pos_match = re.search(r"(\d+)", nuc)
        nt_pos = int(pos_match.group(1)) if pos_match else None

        return ClassifiedMutation(
            original=mutation,
            mutation_type=MutationType.INSERTION,
            nucleotide_position=nt_pos,
            inserted_seq=inserted,
        )

    def _classify_deletion(
        self, mutation: Mutation, nuc: str, del_size: int,
    ) -> ClassifiedMutation:
        pos_match = re.search(r"(\d+)", nuc)
        nt_pos = int(pos_match.group(1)) if pos_match else None

        return ClassifiedMutation(
            original=mutation,
            mutation_type=MutationType.DELETION,
            nucleotide_position=nt_pos,
            deleted_length=del_size,
        )

    def _classify_nt_snp(self, mutation: Mutation, nuc: str) -> ClassifiedMutation:
        """Parse nucleotide SNP from c.NNNX>Y notation."""
        match = re.search(r"(\d+)([ATGC])>([ATGC])", nuc, re.IGNORECASE)
        if match:
            return ClassifiedMutation(
                original=mutation,
                mutation_type=MutationType.NUCLEOTIDE_SNP,
                nucleotide_position=int(match.group(1)),
                ref_base=match.group(2).upper(),
                alt_base=match.group(3).upper(),
            )
        return ClassifiedMutation(
            original=mutation,
            mutation_type=MutationType.UNKNOWN,
        )

    @staticmethod
    def _is_promoter_notation(nuc: str) -> bool:
        return bool(re.search(r"c\.\s*-\d+", nuc))

    @staticmethod
    def _parse_deletion_size(nuc: str) -> int:
        match = re.search(r"(\d+)_(\d+)del", nuc)
        if match:
            return int(match.group(2)) - int(match.group(1)) + 1
        match = re.search(r"del([ATGC]*)", nuc, re.IGNORECASE)
        if match and match.group(1):
            return len(match.group(1))
        return 1


# ======================================================================
# Genome storage
# ======================================================================

@dataclass
class GenomeRecord:
    """A loaded reference genome with metadata."""
    accession: str
    sequence: Seq
    length: int
    organism: str = ""
    description: str = ""


class GenomeStore:
    """Load and index one or more reference genomes.

    Supports multi-chromosome / multi-contig references.
    Primary use case: single-chromosome bacterial genomes,
    but handles plasmids and multi-contig assemblies.
    """

    def __init__(self) -> None:
        self.genomes: dict[str, GenomeRecord] = {}
        self._primary: Optional[str] = None

    def load_fasta(self, path: str | Path, primary: bool = True) -> str:
        """Load genome from FASTA. Returns accession of first record."""
        path = Path(path)
        records = list(SeqIO.parse(path, "fasta"))
        if not records:
            raise ValueError(f"No sequences found in {path}")

        first_accession = None
        for rec in records:
            accession = rec.id
            genome = GenomeRecord(
                accession=accession,
                sequence=rec.seq,
                length=len(rec.seq),
                description=rec.description,
            )
            self.genomes[accession] = genome
            logger.info("Loaded genome: %s (%d bp)", accession, genome.length)
            if first_accession is None:
                first_accession = accession

        if primary and first_accession:
            self._primary = first_accession

        return first_accession  # type: ignore[return-value]

    @property
    def primary(self) -> GenomeRecord:
        if self._primary is None:
            raise RuntimeError("No primary genome loaded")
        return self.genomes[self._primary]

    @property
    def primary_accession(self) -> str:
        if self._primary is None:
            raise RuntimeError("No primary genome loaded")
        return self._primary

    def get_sequence(self, accession: Optional[str] = None) -> Seq:
        """Get sequence by accession, or primary if not specified."""
        if accession is None:
            return self.primary.sequence
        return self.genomes[accession].sequence

    def get_length(self, accession: Optional[str] = None) -> int:
        if accession is None:
            return self.primary.length
        return self.genomes[accession].length


# ======================================================================
# Gene record
# ======================================================================

@dataclass
class GeneRecord:
    """Coordinates and metadata for a single gene."""
    name: str
    start: int                        # 0-based, inclusive
    end: int                          # 0-based, exclusive
    strand: Strand
    accession: str = ""               # which contig/chromosome
    locus_tag: str = ""
    product: str = ""                 # gene product description
    gene_biotype: str = "protein_coding"  # protein_coding, rRNA, tRNA

    @property
    def length_bp(self) -> int:
        return self.end - self.start

    @property
    def length_codons(self) -> int:
        return self.length_bp // 3

    @property
    def is_rrna(self) -> bool:
        return self.gene_biotype in ("rRNA", "rrna") or self.name.lower() in (
            "rrs", "rrl", "rrf", "rrn",
        )

    def codon_to_genomic(self, codon_number: int) -> int:
        """Convert 1-based codon number to genomic nucleotide position."""
        if self.strand == Strand.PLUS:
            return self.start + (codon_number - 1) * 3
        else:
            return self.end - codon_number * 3

    def nucleotide_to_genomic(self, nt_position: int) -> int:
        """Convert 1-based gene-relative nucleotide position to genomic.

        Used for rRNA mutations (A1401G) and nucleotide-level annotations.
        """
        if self.strand == Strand.PLUS:
            return self.start + (nt_position - 1)
        else:
            return self.end - nt_position

    def promoter_to_genomic(self, upstream_distance: int) -> int:
        """Convert promoter distance (positive int) to genomic position.

        upstream_distance=15 means 15 bp upstream of gene start.
        """
        if self.strand == Strand.PLUS:
            return self.start - upstream_distance
        else:
            return self.end + upstream_distance - 1


# ======================================================================
# Codon offset resolution
# ======================================================================

@dataclass
class OffsetResult:
    """Result of offset resolution attempt."""
    success: bool
    genomic_pos: int = 0
    codon_pos_used: int = 0
    offset_applied: int = 0
    ref_codon: str = ""
    strategy: str = ""


class OffsetResolver:
    """5-strategy cascade for codon numbering correction.

    1. Clinical alias table (hardcoded common mutations)
    2. Direct position (assumes native numbering)
    3. Known offsets per gene (E. coli, etc.)
    4. Cached offsets (learned from previous mutations in same gene)
    5. Brute-force scan (search +/-N codons for matching amino acid)

    Organism-configurable: pass your own alias table and offset dict
    for non-TB organisms.
    """

    DEFAULT_KNOWN_OFFSETS: dict[str, list[int]] = {
        "rpoB": [0, 81],
        "rpoC": [0],
        "katG": [0],
        "inhA": [0],
        "embB": [0],
        "embC": [0],
        "gyrA": [0, 5],
        "gyrB": [0],
        "rpsL": [0],
        "rrs":  [0],
        "pncA": [0],
        "ethA": [0],
        "eis":  [0],
        "tlyA": [0],
        "Rv0678": [0],
        "mmpR5": [0],
        "pepQ": [0],
        "ddn":  [0],
        "fbiA": [0],
        "fbiB": [0],
        "fbiC": [0],
        "fgd1": [0],
        "rplC": [0],
        "folC": [0],
        "thyA": [0],
    }

    DEFAULT_CLINICAL_ALIASES: dict[tuple[str, int], int] = {
        ("rpoB", 531): 450,
        ("rpoB", 526): 445,
        ("rpoB", 516): 435,
        ("rpoB", 513): 432,
        ("rpoB", 533): 452,
        ("rpoB", 522): 441,
        ("rpoB", 511): 430,
        ("rpoB", 512): 431,
        ("rpoB", 515): 434,
        ("rpoB", 529): 448,
    }

    def __init__(
        self,
        genome_store: GenomeStore,
        known_offsets: Optional[dict[str, list[int]]] = None,
        clinical_aliases: Optional[dict[tuple[str, int], int]] = None,
        scan_radius: int = 200,
    ) -> None:
        self.store = genome_store
        self.known_offsets = known_offsets or self.DEFAULT_KNOWN_OFFSETS
        self.clinical_aliases = clinical_aliases or self.DEFAULT_CLINICAL_ALIASES
        self.scan_radius = scan_radius
        self._cache: dict[str, int] = {}
        self.stats: dict[str, int] = {
            "alias": 0, "direct": 0, "known_offset": 0,
            "cached": 0, "bruteforce": 0, "failed": 0,
        }

    def resolve_codon(
        self,
        gene: GeneRecord,
        codon_number: int,
        expected_aa: str,
    ) -> OffsetResult:
        """Resolve a codon position using the 5-strategy cascade."""
        seq = self.store.get_sequence(gene.accession or None)

        # Strategy 1: Clinical alias
        alias_key = (gene.name, codon_number)
        if alias_key in self.clinical_aliases:
            mtb_pos = self.clinical_aliases[alias_key]
            result = self._try_position(gene, mtb_pos, expected_aa, seq)
            if result is not None:
                self.stats["alias"] += 1
                logger.info(
                    "Resolved %s codon %d via clinical alias -> %d",
                    gene.name, codon_number, mtb_pos,
                )
                return OffsetResult(
                    success=True, genomic_pos=result[0],
                    codon_pos_used=mtb_pos,
                    offset_applied=codon_number - mtb_pos,
                    ref_codon=result[1], strategy="clinical_alias",
                )

        # Strategy 2: Direct position
        result = self._try_position(gene, codon_number, expected_aa, seq)
        if result is not None:
            self.stats["direct"] += 1
            return OffsetResult(
                success=True, genomic_pos=result[0],
                codon_pos_used=codon_number, offset_applied=0,
                ref_codon=result[1], strategy="direct",
            )

        # Strategy 3: Known offsets
        offsets = self.known_offsets.get(gene.name, [])
        for offset in offsets:
            if offset == 0:
                continue
            adjusted = codon_number - offset
            if adjusted < 1 or adjusted > gene.length_codons:
                continue
            result = self._try_position(gene, adjusted, expected_aa, seq)
            if result is not None:
                self.stats["known_offset"] += 1
                self._cache[gene.name] = offset
                logger.info(
                    "Resolved %s codon %d with known offset %d -> %d",
                    gene.name, codon_number, offset, adjusted,
                )
                return OffsetResult(
                    success=True, genomic_pos=result[0],
                    codon_pos_used=adjusted, offset_applied=offset,
                    ref_codon=result[1],
                    strategy=f"known_offset_{offset}",
                )

        # Strategy 4: Cached offset
        if gene.name in self._cache:
            cached = self._cache[gene.name]
            adjusted = codon_number - cached
            if 1 <= adjusted <= gene.length_codons:
                result = self._try_position(gene, adjusted, expected_aa, seq)
                if result is not None:
                    self.stats["cached"] += 1
                    return OffsetResult(
                        success=True, genomic_pos=result[0],
                        codon_pos_used=adjusted, offset_applied=cached,
                        ref_codon=result[1],
                        strategy=f"cached_offset_{cached}",
                    )

        # Strategy 5: Brute-force scan
        scan = self._scan_for_aa(gene, codon_number, expected_aa, seq)
        if scan is not None:
            genomic_pos, actual_pos, ref_codon = scan
            inferred_offset = codon_number - actual_pos
            self._cache[gene.name] = inferred_offset
            self.stats["bruteforce"] += 1
            logger.warning(
                "Brute-force resolved %s codon %d: inferred offset %d -> %d. "
                "Consider adding to known_offsets.",
                gene.name, codon_number, inferred_offset, actual_pos,
            )
            return OffsetResult(
                success=True, genomic_pos=genomic_pos,
                codon_pos_used=actual_pos, offset_applied=inferred_offset,
                ref_codon=ref_codon,
                strategy=f"bruteforce_offset_{inferred_offset}",
            )

        self.stats["failed"] += 1
        return OffsetResult(success=False)

    def _try_position(
        self, gene: GeneRecord, codon_pos: int, expected_aa: str, seq: Seq,
    ) -> Optional[tuple[int, str]]:
        if codon_pos < 1 or codon_pos > gene.length_codons:
            return None
        genomic_pos = gene.codon_to_genomic(codon_pos)
        if genomic_pos < 0 or genomic_pos + 3 > len(seq):
            return None
        ref_codon = self._extract_codon(seq, genomic_pos, gene.strand)
        aa = str(Seq(ref_codon).translate())
        if aa == expected_aa:
            return genomic_pos, ref_codon
        return None

    def _scan_for_aa(
        self, gene: GeneRecord, center: int, expected_aa: str, seq: Seq,
    ) -> Optional[tuple[int, int, str]]:
        for delta in range(1, self.scan_radius + 1):
            for candidate in [center - delta, center + delta]:
                if candidate < 1 or candidate > gene.length_codons:
                    continue
                genomic_pos = gene.codon_to_genomic(candidate)
                if genomic_pos < 0 or genomic_pos + 3 > len(seq):
                    continue
                ref_codon = self._extract_codon(seq, genomic_pos, gene.strand)
                aa = str(Seq(ref_codon).translate())
                if aa == expected_aa:
                    return genomic_pos, candidate, ref_codon
        return None

    @staticmethod
    def _extract_codon(seq: Seq, genomic_pos: int, strand: Strand) -> str:
        codon = str(seq[genomic_pos : genomic_pos + 3])
        if strand == Strand.MINUS:
            codon = str(Seq(codon).reverse_complement())
        return codon.upper()

    def summary(self) -> str:
        total = sum(self.stats.values())
        if total == 0:
            return "OffsetResolver: no resolutions attempted"
        lines = [f"OffsetResolver: {total} resolutions attempted"]
        for strategy, count in self.stats.items():
            if count > 0:
                lines.append(
                    f"  {strategy:20s}: {count:4d} ({count/total:.0%})"
                )
        return "\n".join(lines)


# ======================================================================
# Annotation parser
# ======================================================================

class AnnotationParser:
    """Parse gene annotations from GFF3 or GenBank format."""

    @staticmethod
    def parse_gff3(
        gff_path: Path,
        default_accession: str = "",
    ) -> dict[str, GeneRecord]:
        """Parse GFF3 for gene features."""
        genes: dict[str, GeneRecord] = {}

        with open(gff_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 9:
                    continue

                seqid = parts[0]
                feat_type = parts[2]
                start = int(parts[3]) - 1       # GFF 1-based -> 0-based
                end = int(parts[4])              # GFF inclusive -> exclusive
                strand_str = parts[6]
                attrs_str = parts[8]

                attrs = {}
                for kv in attrs_str.split(";"):
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        attrs[k] = v

                strand = Strand.PLUS if strand_str == "+" else Strand.MINUS

                if feat_type == "gene":
                    name = attrs.get("Name", attrs.get("gene", ""))
                    locus_tag = attrs.get(
                        "locus_tag", attrs.get("old_locus_tag", ""),
                    )
                    biotype = attrs.get("gene_biotype", "protein_coding")

                    if not name:
                        name = locus_tag
                    if not name:
                        continue

                    accession = default_accession or seqid

                    gene = GeneRecord(
                        name=name,
                        start=start,
                        end=end,
                        strand=strand,
                        accession=accession,
                        locus_tag=locus_tag,
                        gene_biotype=biotype,
                    )
                    genes[name] = gene

                    # Also index by locus_tag
                    if locus_tag and locus_tag != name:
                        genes[locus_tag] = gene

                elif feat_type == "CDS":
                    gene_name = attrs.get("gene", "")
                    product = attrs.get("product", "")
                    if gene_name and gene_name in genes and product:
                        genes[gene_name].product = product

        logger.info("Parsed %d genes from GFF3", len(genes))
        return genes

    @staticmethod
    def parse_genbank(gb_path: Path) -> dict[str, GeneRecord]:
        """Parse GenBank format for gene features."""
        genes: dict[str, GeneRecord] = {}
        for record in SeqIO.parse(gb_path, "genbank"):
            accession = record.id
            for feature in record.features:
                if feature.type not in ("gene", "CDS", "rRNA"):
                    continue
                qualifiers = feature.qualifiers
                name = qualifiers.get(
                    "gene", qualifiers.get("locus_tag", [""]),
                )[0]
                if not name:
                    continue

                start = int(feature.location.start)
                end = int(feature.location.end)
                strand = (
                    Strand.PLUS
                    if feature.location.strand == 1
                    else Strand.MINUS
                )
                locus_tag = qualifiers.get("locus_tag", [""])[0]
                product = qualifiers.get("product", [""])[0]
                biotype = "rRNA" if feature.type == "rRNA" else "protein_coding"

                gene = GeneRecord(
                    name=name,
                    start=start,
                    end=end,
                    strand=strand,
                    accession=accession,
                    locus_tag=locus_tag,
                    product=product,
                    gene_biotype=biotype,
                )
                genes[name] = gene
                if locus_tag and locus_tag != name:
                    genes[locus_tag] = gene

        logger.info("Parsed %d genes from GenBank", len(genes))
        return genes


# ======================================================================
# Validation engine
# ======================================================================

@dataclass
class ValidationResult:
    """Cross-check results for a resolved target."""
    valid: bool
    checks: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        status = "PASS" if self.valid else "FAIL"
        failed = [k for k, v in self.checks.items() if not v]
        msg = f"Validation: {status}"
        if failed:
            msg += f" (failed: {', '.join(failed)})"
        if self.warnings:
            msg += f" [warnings: {'; '.join(self.warnings)}]"
        return msg


class TargetValidator:
    """Cross-check resolved targets for consistency.

    Checks:
    1. ref_codon translates to ref_aa
    2. alt_codon translates to alt_aa
    3. ref and alt differ by exactly 1 nt (for SNPs)
    4. genomic_pos within gene boundaries
    5. flanking sequence matches genome
    """

    def validate(
        self, target: Target, gene: GeneRecord, seq: Seq,
    ) -> ValidationResult:
        checks: dict[str, bool] = {}
        warnings: list[str] = []
        mutation = target.mutation

        # Check 1: ref_codon -> ref_aa
        if target.ref_codon and len(target.ref_codon) == 3:
            expected_ref = str(Seq(target.ref_codon).translate())
            checks["ref_codon_aa"] = expected_ref == mutation.ref_aa
            if not checks["ref_codon_aa"]:
                warnings.append(
                    f"ref_codon {target.ref_codon} -> {expected_ref}, "
                    f"expected {mutation.ref_aa}"
                )

        # Check 2: alt_codon -> alt_aa
        if (
            target.alt_codon
            and len(target.alt_codon) == 3
            and target.alt_codon != target.ref_codon
        ):
            expected_alt = str(Seq(target.alt_codon).translate())
            checks["alt_codon_aa"] = expected_alt == mutation.alt_aa
            if not checks["alt_codon_aa"]:
                warnings.append(
                    f"alt_codon {target.alt_codon} -> {expected_alt}, "
                    f"expected {mutation.alt_aa}"
                )

        # Check 3: single-nt change (codon-level mutations only)
        if (
            target.ref_codon
            and target.alt_codon
            and len(target.ref_codon) == 3
            and len(target.alt_codon) == 3
        ):
            diffs = sum(
                a != b for a, b in zip(target.ref_codon, target.alt_codon)
            )
            checks["single_nt_change"] = diffs == 1
            if diffs != 1:
                warnings.append(
                    f"ref->alt requires {diffs} nt changes "
                    f"({target.ref_codon}->{target.alt_codon})"
                )

        # Check 4: position within gene (skip for promoter mutations)
        if not getattr(mutation, "position", 0) < 0:
            checks["within_gene"] = gene.start <= target.genomic_pos < gene.end
            if not checks["within_gene"]:
                warnings.append(
                    f"genomic_pos {target.genomic_pos} outside gene "
                    f"[{gene.start}, {gene.end})"
                )

        # Check 5: flanking matches genome
        flank_start = target.flanking_start
        expected_flank = str(
            seq[flank_start : flank_start + len(target.flanking_seq)]
        )
        checks["flanking_match"] = expected_flank == target.flanking_seq
        if not checks["flanking_match"]:
            warnings.append(
                "flanking_seq does not match genome at claimed position"
            )

        valid = all(checks.values())
        return ValidationResult(valid=valid, checks=checks, warnings=warnings)


# ======================================================================
# Main resolver — public API
# ======================================================================

class TargetResolver:
    """Resolve clinical mutations to genomic targets.

    Organism-agnostic, mutation-type-complete.

    Usage:
        resolver = TargetResolver(
            fasta="data/references/H37Rv.fasta",
            gff="data/references/H37Rv.gff3",
        )
        targets = resolver.resolve_all(mutations)

        # Non-TB organism:
        resolver = TargetResolver(
            fasta="my_genome.fasta",
            gff="my_genome.gff3",
            known_offsets={"gyrA": [0, 3]},
            clinical_aliases={},
        )

        # With validation:
        targets = resolver.resolve_all(mutations, validate=True)
    """

    def __init__(
        self,
        fasta: str | Path,
        gff: Optional[str | Path] = None,
        genbank: Optional[str | Path] = None,
        gene_table: Optional[dict[str, GeneRecord]] = None,
        known_offsets: Optional[dict[str, list[int]]] = None,
        clinical_aliases: Optional[dict[tuple[str, int], int]] = None,
        flanking_window: int = FLANKING_WINDOW,
        default_accession: str = H37RV_ACCESSION,
        scan_radius: int = 200,
    ) -> None:
        # Load genome
        self.genome_store = GenomeStore()
        self.genome_store.load_fasta(fasta)
        self.default_accession = self.genome_store.primary_accession
        self.flanking_window = flanking_window

        # Load annotation
        if gene_table is not None:
            self.gene_table = gene_table
        elif gff is not None:
            self.gene_table = AnnotationParser.parse_gff3(
                Path(gff), self.default_accession,
            )
        elif genbank is not None:
            self.gene_table = AnnotationParser.parse_genbank(Path(genbank))
        else:
            self.gene_table = {}
            logger.warning("No annotation provided — gene lookup will fail")

        # Sub-components
        self.classifier = MutationClassifier()
        self.offset_resolver = OffsetResolver(
            genome_store=self.genome_store,
            known_offsets=known_offsets,
            clinical_aliases=clinical_aliases,
            scan_radius=scan_radius,
        )
        self.validator = TargetValidator()

    def resolve(
        self, mutation: Mutation, validate: bool = False,
    ) -> Optional[Target]:
        """Resolve a single mutation to a genomic Target."""
        classified = self.classifier.classify(mutation)
        gene = self._find_gene(mutation.gene)

        if gene is None:
            logger.warning("Gene %s not found in annotation", mutation.gene)
            return None

        target: Optional[Target] = None

        if classified.mutation_type == MutationType.AA_SUBSTITUTION:
            target = self._resolve_aa_substitution(mutation, gene)

        elif classified.mutation_type == MutationType.NUCLEOTIDE_SNP:
            target = self._resolve_nt_snp(mutation, gene, classified)

        elif classified.mutation_type == MutationType.RRNA:
            target = self._resolve_rrna(mutation, gene, classified)

        elif classified.mutation_type == MutationType.PROMOTER:
            target = self._resolve_promoter(mutation, gene, classified)

        elif classified.mutation_type in (
            MutationType.INSERTION, MutationType.DELETION,
        ):
            target = self._resolve_indel(mutation, gene, classified)

        elif classified.mutation_type == MutationType.LARGE_DELETION:
            target = self._resolve_large_deletion(mutation, gene)

        elif classified.mutation_type in (
            MutationType.FRAMESHIFT, MutationType.MNV, MutationType.UNKNOWN,
        ):
            target = self._resolve_aa_substitution(mutation, gene)

        if target is not None and validate:
            result = self.validator.validate(
                target, gene, self.genome_store.get_sequence(),
            )
            if not result.valid:
                logger.warning(
                    "Validation failed for %s: %s",
                    mutation.label, result.summary,
                )

        return target

    def resolve_all(
        self, mutations: list[Mutation], validate: bool = False,
    ) -> list[Target]:
        """Resolve a batch. Skips unresolvable entries."""
        targets = []
        for mut in mutations:
            t = self.resolve(mut, validate=validate)
            if t is not None:
                targets.append(t)

        logger.info(
            "Resolved %d / %d mutations", len(targets), len(mutations),
        )
        if self.offset_resolver.stats["failed"] > 0:
            logger.info(self.offset_resolver.summary())

        return targets

    # ------------------------------------------------------------------
    # Resolution strategies
    # ------------------------------------------------------------------

    def _resolve_aa_substitution(
        self, mutation: Mutation, gene: GeneRecord,
    ) -> Optional[Target]:
        """Standard amino acid substitution (e.g. S450L)."""
        result = self.offset_resolver.resolve_codon(
            gene, mutation.position, mutation.ref_aa,
        )
        if not result.success:
            logger.error(
                "Cannot resolve %s: ref_aa %s not found at position %d "
                "in gene %s (%d codons, strand %s)",
                mutation.label, mutation.ref_aa, mutation.position,
                gene.name, gene.length_codons, gene.strand.value,
            )
            return None

        alt_codon = self._infer_alt_codon(result.ref_codon, mutation.alt_aa)
        return self._build_target(
            mutation, gene, result.genomic_pos, result.ref_codon, alt_codon,
        )

    def _resolve_nt_snp(
        self,
        mutation: Mutation,
        gene: GeneRecord,
        classified: ClassifiedMutation,
    ) -> Optional[Target]:
        """Nucleotide-level SNP within CDS (e.g. c.1349C>T)."""
        nt_pos = classified.nucleotide_position
        if nt_pos is None:
            return self._resolve_aa_substitution(mutation, gene)

        genomic_pos = gene.nucleotide_to_genomic(nt_pos)
        seq = self.genome_store.get_sequence()

        # Snap to codon boundary
        if gene.strand == Strand.PLUS:
            codon_start = gene.start + ((genomic_pos - gene.start) // 3) * 3
        else:
            codon_start = gene.end - (((gene.end - genomic_pos) // 3) + 1) * 3

        ref_codon = self._extract_codon(seq, codon_start, gene.strand)

        # Build alt codon
        if classified.ref_base and classified.alt_base:
            if gene.strand == Strand.PLUS:
                offset_in_codon = genomic_pos - codon_start
            else:
                offset_in_codon = codon_start + 2 - genomic_pos
            alt_list = list(ref_codon)
            if 0 <= offset_in_codon < 3:
                alt_list[offset_in_codon] = classified.alt_base
            alt_codon = "".join(alt_list)
        else:
            alt_codon = ref_codon

        return self._build_target(
            mutation, gene, codon_start, ref_codon, alt_codon,
        )

    def _resolve_rrna(
        self,
        mutation: Mutation,
        gene: GeneRecord,
        classified: ClassifiedMutation,
    ) -> Optional[Target]:
        """rRNA mutation (e.g. rrs A1401G)."""
        nt_pos = classified.nucleotide_position or mutation.position
        genomic_pos = gene.nucleotide_to_genomic(nt_pos)
        seq = self.genome_store.get_sequence()

        if genomic_pos < 0 or genomic_pos >= len(seq):
            logger.error("rRNA position %d out of genome bounds", nt_pos)
            return None

        ref_base = str(seq[genomic_pos]).upper()
        if gene.strand == Strand.MINUS:
            ref_base = str(Seq(ref_base).complement())

        if classified.ref_base and ref_base != classified.ref_base:
            logger.warning(
                "rRNA %s: expected %s at position %d, found %s",
                mutation.label, classified.ref_base, nt_pos, ref_base,
            )

        alt_base = classified.alt_base or mutation.alt_aa

        flank_start = max(0, genomic_pos - self.flanking_window)
        flank_end = min(len(seq), genomic_pos + 1 + self.flanking_window)
        flanking = str(seq[flank_start:flank_end])

        return Target(
            mutation=mutation,
            chrom=self.default_accession,
            genomic_pos=genomic_pos,
            ref_codon=ref_base,
            alt_codon=alt_base,
            flanking_seq=flanking,
            flanking_start=flank_start,
        )

    def _resolve_promoter(
        self,
        mutation: Mutation,
        gene: GeneRecord,
        classified: ClassifiedMutation,
    ) -> Optional[Target]:
        """Promoter mutation (e.g. inhA c.-15C>T)."""
        distance = classified.promoter_distance or abs(mutation.position)
        genomic_pos = gene.promoter_to_genomic(distance)
        seq = self.genome_store.get_sequence()

        if genomic_pos < 0 or genomic_pos >= len(seq):
            logger.error("Promoter position -%d out of bounds", distance)
            return None

        ref_base = str(seq[genomic_pos]).upper()
        if gene.strand == Strand.MINUS:
            ref_base = str(Seq(ref_base).complement())

        alt_base = classified.alt_base or mutation.alt_aa

        flank_start = max(0, genomic_pos - self.flanking_window)
        flank_end = min(len(seq), genomic_pos + 1 + self.flanking_window)
        flanking = str(seq[flank_start:flank_end])

        return Target(
            mutation=mutation,
            chrom=self.default_accession,
            genomic_pos=genomic_pos,
            ref_codon=ref_base,
            alt_codon=alt_base,
            flanking_seq=flanking,
            flanking_start=flank_start,
        )

    def _resolve_indel(
        self,
        mutation: Mutation,
        gene: GeneRecord,
        classified: ClassifiedMutation,
    ) -> Optional[Target]:
        """Insertion or deletion within CDS.

        Extracts flanking context around the indel site for crRNA design.
        The crRNA can target either WT (absence detection) or mutant
        (presence detection) sequence.
        """
        nt_pos = classified.nucleotide_position
        if nt_pos is None:
            nt_pos = (mutation.position - 1) * 3 + 1

        genomic_pos = gene.nucleotide_to_genomic(nt_pos)
        seq = self.genome_store.get_sequence()

        if genomic_pos < 0 or genomic_pos >= len(seq):
            logger.error("Indel position %d out of bounds", nt_pos)
            return None

        context_len = max(6, (classified.deleted_length or 1) + 3)
        ref_context = str(seq[genomic_pos : genomic_pos + context_len]).upper()

        if classified.mutation_type == MutationType.INSERTION:
            inserted = classified.inserted_seq or "N"
            alt_context = ref_context[:3] + inserted + ref_context[3:]
        else:
            del_len = classified.deleted_length or 1
            alt_context = ref_context[:3] + ref_context[3 + del_len:]

        flank_start = max(0, genomic_pos - self.flanking_window)
        flank_end = min(
            len(seq), genomic_pos + context_len + self.flanking_window,
        )
        flanking = str(seq[flank_start:flank_end])

        return Target(
            mutation=mutation,
            chrom=self.default_accession,
            genomic_pos=genomic_pos,
            ref_codon=ref_context[:3],
            alt_codon=alt_context[:3],
            flanking_seq=flanking,
            flanking_start=flank_start,
        )

    def _resolve_large_deletion(
        self, mutation: Mutation, gene: GeneRecord,
    ) -> Optional[Target]:
        """Large deletion / full gene deletion.

        Targets the middle of the gene for presence/absence assay.
        """
        seq = self.genome_store.get_sequence()
        mid = (gene.start + gene.end) // 2
        flank_start = max(0, mid - self.flanking_window)
        flank_end = min(len(seq), mid + self.flanking_window)
        flanking = str(seq[flank_start:flank_end])
        ref_codon = str(seq[mid : mid + 3]).upper()

        return Target(
            mutation=mutation,
            chrom=self.default_accession,
            genomic_pos=mid,
            ref_codon=ref_codon,
            alt_codon="---",
            flanking_seq=flanking,
            flanking_start=flank_start,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_gene(self, gene_name: str) -> Optional[GeneRecord]:
        """Look up gene by name, locus_tag, or common aliases."""
        if gene_name in self.gene_table:
            return self.gene_table[gene_name]

        # Case-insensitive
        for name, rec in self.gene_table.items():
            if name.lower() == gene_name.lower():
                return rec

        # Known aliases
        ALIASES: dict[str, str] = {
            "mmpR5": "Rv0678",
            "Rv0678": "mmpR5",
            "fabG1": "inhA",
        }
        alias = ALIASES.get(gene_name)
        if alias and alias in self.gene_table:
            return self.gene_table[alias]

        return None

    def _build_target(
        self,
        mutation: Mutation,
        gene: GeneRecord,
        genomic_pos: int,
        ref_codon: str,
        alt_codon: str,
    ) -> Target:
        seq = self.genome_store.get_sequence()
        flank_start = max(0, genomic_pos - self.flanking_window)
        flank_end = min(len(seq), genomic_pos + 3 + self.flanking_window)
        flanking = str(seq[flank_start:flank_end])

        return Target(
            mutation=mutation,
            chrom=self.default_accession,
            genomic_pos=genomic_pos,
            ref_codon=ref_codon,
            alt_codon=alt_codon,
            flanking_seq=flanking,
            flanking_start=flank_start,
        )

    @staticmethod
    def _extract_codon(seq: Seq, genomic_pos: int, strand: Strand) -> str:
        codon = str(seq[genomic_pos : genomic_pos + 3])
        if strand == Strand.MINUS:
            codon = str(Seq(codon).reverse_complement())
        return codon.upper()

    @staticmethod
    def _infer_alt_codon(ref_codon: str, alt_aa: str) -> str:
        """Find single-nt change producing alt_aa. Prefer transitions."""
        transitions = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
        candidates: list[tuple[str, bool]] = []

        for i in range(3):
            for nt in "ATGC":
                if nt == ref_codon[i]:
                    continue
                alt = ref_codon[:i] + nt + ref_codon[i + 1:]
                aa = str(Seq(alt).translate())
                if aa == alt_aa:
                    is_transition = (ref_codon[i], nt) in transitions
                    candidates.append((alt, is_transition))

        if not candidates:
            logger.warning(
                "No single-nt change produces %s from %s", alt_aa, ref_codon,
            )
            return ref_codon

        candidates.sort(key=lambda x: not x[1])
        return candidates[0][0]
