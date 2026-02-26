#!/usr/bin/env python
"""Quick design script — design crRNAs for the core MDR-TB targets.

Usage:
    python scripts/design_core_panel.py --reference data/references/H37Rv.fasta
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from saber.core.config import PipelineConfig, ReferenceConfig
from saber.core.types import Drug, Mutation
from saber.pipeline.runner import SABERPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Core MDR-TB targets (>95% clinical resistance coverage)
CORE_TARGETS = [
    Mutation(gene="rpoB", position=531, ref_aa="S", alt_aa="L", drug=Drug.RIFAMPICIN),
    Mutation(gene="rpoB", position=526, ref_aa="H", alt_aa="Y", drug=Drug.RIFAMPICIN),
    Mutation(gene="rpoB", position=516, ref_aa="D", alt_aa="V", drug=Drug.RIFAMPICIN),
    Mutation(gene="katG", position=315, ref_aa="S", alt_aa="T", drug=Drug.ISONIAZID),
    Mutation(gene="inhA", position=15, ref_aa="C", alt_aa="T", drug=Drug.ISONIAZID),   # promoter
    Mutation(gene="embB", position=306, ref_aa="M", alt_aa="V", drug=Drug.ETHAMBUTOL),
    Mutation(gene="embB", position=406, ref_aa="G", alt_aa="A", drug=Drug.ETHAMBUTOL),
    Mutation(gene="gyrA", position=94, ref_aa="D", alt_aa="G", drug=Drug.FLUOROQUINOLONE),
    Mutation(gene="gyrA", position=90, ref_aa="A", alt_aa="V", drug=Drug.FLUOROQUINOLONE),
    Mutation(gene="rpsL", position=43, ref_aa="K", alt_aa="R", drug=Drug.AMINOGLYCOSIDE),
    Mutation(gene="pncA", position=8, ref_aa="D", alt_aa="N", drug=Drug.PYRAZINAMIDE),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--gff", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/core_panel"))
    args = parser.parse_args()

    config = PipelineConfig(
        name="core_mdr_panel",
        output_dir=args.output,
        reference=ReferenceConfig(
            genome_fasta=args.reference,
            gff_annotation=args.gff,
        ),
    )

    pipeline = SABERPipeline(config)
    results = pipeline.run(CORE_TARGETS)

    # Summary
    print(f"\n{'='*70}")
    print(f"SABER Core Panel Design Summary")
    print(f"{'='*70}")
    for label, scored in results.items():
        if scored:
            top = scored[0]
            print(f"  {label:20s}  top={top.candidate.spacer_seq}  score={top.heuristic.composite:.3f}  n={len(scored)}")
        else:
            print(f"  {label:20s}  NO CANDIDATES FOUND")


if __name__ == "__main__":
    main()
