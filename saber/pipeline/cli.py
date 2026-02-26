"""Command-line interface for SABER.

Usage:
    saber design --gene rpoB --mutation S531L --reference H37Rv.fasta
    saber panel --config configs/mdr_14plex.yaml
    saber score --candidates results/candidates.json --model checkpoints/jepa.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from saber.core.config import PipelineConfig
from saber.core.types import Drug, Mutation


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="saber",
        description="SABER — crRNA design for CRISPR-Cas12a TB diagnostics",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    # -- design: single target --
    p_design = sub.add_parser("design", help="Design crRNAs for a single target")
    p_design.add_argument("--gene", required=True)
    p_design.add_argument("--mutation", required=True, help="e.g. S531L")
    p_design.add_argument("--drug", default="RIF")
    p_design.add_argument("--reference", required=True, type=Path)
    p_design.add_argument("--gff", type=Path, default=None)
    p_design.add_argument("--output", type=Path, default=Path("results"))

    # -- panel: full multiplex --
    p_panel = sub.add_parser("panel", help="Design full multiplex panel from config")
    p_panel.add_argument("--config", required=True, type=Path)

    # -- score: score pre-generated candidates --
    p_score = sub.add_parser("score", help="Score candidates with ML model")
    p_score.add_argument("--candidates", required=True, type=Path)
    p_score.add_argument("--model", required=True, type=Path)
    p_score.add_argument("--output", type=Path, default=Path("results"))

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "design":
        _cmd_design(args)
    elif args.command == "panel":
        _cmd_panel(args)
    elif args.command == "score":
        _cmd_score(args)


def _cmd_design(args: argparse.Namespace) -> None:
    """Design crRNAs for a single gene:mutation target."""
    import re
    match = re.match(r"([A-Z])(\d+)([A-Z])", args.mutation)
    if not match:
        print(f"Error: invalid mutation format '{args.mutation}'. Expected e.g. S531L", file=sys.stderr)
        sys.exit(1)

    mutation = Mutation(
        gene=args.gene,
        position=int(match.group(2)),
        ref_aa=match.group(1),
        alt_aa=match.group(3),
        drug=Drug(args.drug),
    )

    from saber.core.config import ReferenceConfig
    config = PipelineConfig(
        name=f"design_{mutation.label}",
        output_dir=args.output,
        reference=ReferenceConfig(
            genome_fasta=args.reference,
            gff_annotation=args.gff,
        ),
    )

    from saber.pipeline.runner import SABERPipeline
    pipeline = SABERPipeline(config)
    results = pipeline.run([mutation])

    for label, scored in results.items():
        print(f"\n{'='*60}")
        print(f"Target: {label} — {len(scored)} candidates")
        print(f"{'='*60}")
        for s in scored[:10]:
            c = s.candidate
            print(
                f"  #{s.rank:2d}  {c.candidate_id}  "
                f"spacer={c.spacer_seq}  "
                f"PAM={c.pam_seq}({c.pam_variant.value})  "
                f"seed_pos={c.mutation_position_in_spacer}  "
                f"score={s.heuristic.composite:.3f}"
            )


def _cmd_panel(args: argparse.Namespace) -> None:
    """Design full multiplex panel from YAML config."""
    config = PipelineConfig.from_yaml(args.config)

    from saber.targets.who_parser import WHOCatalogueParser
    parser = WHOCatalogueParser(path="data/who_catalogue/catalogue_v2.csv")
    mutations = parser.parse()

    from saber.pipeline.runner import SABERPipeline
    pipeline = SABERPipeline(config)
    results = pipeline.run(mutations)
    print(f"\nPanel design complete. {sum(len(v) for v in results.values())} total candidates.")


def _cmd_score(args: argparse.Namespace) -> None:
    """Re-score candidates with a trained ML model."""
    print(f"Scoring candidates from {args.candidates} with model {args.model}")
    # Implementation: load candidates JSON, init JEPA scorer, re-rank
    print("Not yet implemented — requires trained model checkpoint.")


if __name__ == "__main__":
    main()
