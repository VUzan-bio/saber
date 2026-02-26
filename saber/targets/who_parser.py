"""Parse the WHO 2023 mutation catalogue into structured Mutation objects.

The catalogue is distributed as an Excel/TSV file with columns for gene, mutation,
drug, confidence grading, etc. This parser normalises the heterogeneous notation
into the canonical form used throughout SABER.

Reference: WHO (2023). Catalogue of mutations in Mycobacterium tuberculosis complex
and their association with drug resistance, 2nd edition.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from saber.core.types import Drug, Mutation

logger = logging.getLogger(__name__)

# WHO catalogue drug name → Drug enum
_DRUG_MAP: dict[str, Drug] = {
    "isoniazid": Drug.ISONIAZID,
    "rifampicin": Drug.RIFAMPICIN,
    "ethambutol": Drug.ETHAMBUTOL,
    "pyrazinamide": Drug.PYRAZINAMIDE,
    "levofloxacin": Drug.FLUOROQUINOLONE,
    "moxifloxacin": Drug.FLUOROQUINOLONE,
    "amikacin": Drug.AMINOGLYCOSIDE,
    "kanamycin": Drug.AMINOGLYCOSIDE,
    "bedaquiline": Drug.BEDAQUILINE,
    "linezolid": Drug.LINEZOLID,
}

# Pattern: S531L, D516V, etc.
_AA_MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
# Pattern: c.1349C>T
_NT_CHANGE_RE = re.compile(r"c\.(\d+)([ATGC])>([ATGC])")


class WHOCatalogueParser:
    """Parse WHO mutation catalogue into Mutation objects.

    Usage:
        parser = WHOCatalogueParser(path="data/who_catalogue/catalogue_v2.csv")
        mutations = parser.parse()
        rif_mutations = parser.filter_by_drug(Drug.RIFAMPICIN)
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._mutations: Optional[list[Mutation]] = None

    def parse(self) -> list[Mutation]:
        """Load and parse the full catalogue."""
        if self._mutations is not None:
            return self._mutations

        df = self._load_dataframe()
        mutations: list[Mutation] = []

        for _, row in df.iterrows():
            mut = self._parse_row(row)
            if mut is not None:
                mutations.append(mut)

        logger.info("Parsed %d mutations from WHO catalogue", len(mutations))
        self._mutations = mutations
        return mutations

    def filter_by_drug(self, drug: Drug) -> list[Mutation]:
        """Return mutations associated with a specific drug."""
        return [m for m in self.parse() if m.drug == drug]

    def filter_by_gene(self, gene: str) -> list[Mutation]:
        """Return mutations in a specific gene."""
        return [m for m in self.parse() if m.gene == gene]

    def get_panel_mutations(self, panel: list[str]) -> list[Mutation]:
        """Get mutations matching a list of labels like ['rpoB_S531L', 'katG_S315T']."""
        label_set = set(panel)
        return [m for m in self.parse() if m.label in label_set]

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_dataframe(self) -> pd.DataFrame:
        suffix = self.path.suffix.lower()
        if suffix in (".csv", ".tsv"):
            sep = "\t" if suffix == ".tsv" else ","
            return pd.read_csv(self.path, sep=sep)
        elif suffix in (".xlsx", ".xls"):
            return pd.read_excel(self.path)
        raise ValueError(f"Unsupported file format: {suffix}")

    def _parse_row(self, row: pd.Series) -> Optional[Mutation]:
        """Parse a single catalogue row. Returns None if unparseable."""
        gene = str(row.get("gene", row.get("Gene", ""))).strip()
        mutation_str = str(row.get("mutation", row.get("Mutation", ""))).strip()
        drug_str = str(row.get("drug", row.get("Drug", ""))).strip().lower()

        drug = _DRUG_MAP.get(drug_str)
        if drug is None:
            return None

        # Parse amino acid change
        match = _AA_MUTATION_RE.match(mutation_str)
        if match is None:
            logger.debug("Skipping non-standard mutation: %s %s", gene, mutation_str)
            return None

        ref_aa, position, alt_aa = match.group(1), int(match.group(2)), match.group(3)

        # Extract nucleotide change if present
        nt_change = None
        nt_col = row.get("nucleotide_change", row.get("Nucleotide_change"))
        if pd.notna(nt_col):
            nt_match = _NT_CHANGE_RE.search(str(nt_col))
            if nt_match:
                nt_change = str(nt_col).strip()

        confidence = str(row.get("confidence", row.get("Confidence", ""))).strip()

        return Mutation(
            gene=gene,
            position=position,
            ref_aa=ref_aa,
            alt_aa=alt_aa,
            nucleotide_change=nt_change,
            drug=drug,
            who_confidence=confidence if confidence else "assoc w resistance",
        )
