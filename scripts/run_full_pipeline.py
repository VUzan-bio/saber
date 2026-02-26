#!/usr/bin/env python3
"""Full MDR-TB 14-plex pipeline: design → enhance → visualise.

Runs the complete SABER workflow:
  1. Design crRNA candidates for all 14 WHO-critical resistance mutations
  2. Apply synthetic mismatch enhancement on all direct candidates
  3. Generate publication-quality figures (11 figures, PNG + SVG)

Usage:
    python scripts/run_full_pipeline.py \
        -r data/references/H37Rv.fasta \
        -g data/references/H37Rv.gff3 \
        -o results/mdr_14plex_full

    # Skip re-designing if results already exist:
    python scripts/run_full_pipeline.py \
        -r data/references/H37Rv.fasta \
        -g data/references/H37Rv.gff3 \
        -o results/mdr_14plex_full --skip-design

    # Only regenerate figures:
    python scripts/run_full_pipeline.py \
        -r data/references/H37Rv.fasta \
        -g data/references/H37Rv.gff3 \
        -o results/mdr_14plex_full --skip-design --skip-enhance
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure saber is importable when run from scripts/ or repo root
_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("full_pipeline")


# ======================================================================
# STEP 1: Panel design
# ======================================================================

def step1_design(reference: str, gff: str, output_dir: Path):
    """Design crRNA candidates for 14-plex MDR-TB panel."""
    log.info("\n" + "=" * 70)
    log.info("  STEP 1: DESIGNING 14-PLEX MDR-TB PANEL")
    log.info("=" * 70)

    from scripts.design_core_panel import run_panel
    run_panel(
        reference=reference,
        gff=gff,
        output_dir=str(output_dir),
    )


# ======================================================================
# STEP 2: Synthetic mismatch enhancement
# ======================================================================

def step2_enhance(output_dir: Path) -> dict:
    """Run synthetic mismatch enhancement on all scored candidates.

    Returns dict: target_label → list of EnhancementReport
    """
    log.info("\n" + "=" * 70)
    log.info("  STEP 2: SYNTHETIC MISMATCH ENHANCEMENT")
    log.info("=" * 70)

    from saber.candidates.synthetic_mismatch import (
        generate_enhanced_variants,
        EnhancementConfig,
        EnhancementReport,
    )

    config = EnhancementConfig(
        cas_variant="enAsCas12a",
        allow_double_synthetic=True,
        min_activity_vs_mut=0.25,
        search_radius=6,
    )

    all_reports: dict[str, list[EnhancementReport]] = {}

    for d in sorted(output_dir.iterdir()):
        if not d.is_dir():
            continue
        scored_path = d / "scored_candidates.json"
        if not scored_path.exists():
            continue

        with open(scored_path) as f:
            raw = json.load(f)

        # Handle both dict and list formats
        if isinstance(raw, dict):
            scored = []
            for key, val in raw.items():
                if isinstance(val, list):
                    scored.extend(val)
                else:
                    scored.append(val)
        else:
            scored = raw

        if not scored:
            continue

        target_label = d.name
        reports: list[EnhancementReport] = []
        n_direct = 0
        n_enhanced = 0

        for sc in scored:
            cand = sc.get("candidate", sc)
            cid = cand.get("candidate_id", "unknown")
            mm_pos = cand.get("mutation_position_in_spacer")

            if not mm_pos or mm_pos < 1:
                continue  # proximity — skip

            n_direct += 1
            spacer = cand["spacer_seq"]
            mut_seq = spacer

            # Approximate WT by flipping the mutation base (transition)
            wt_seq = list(spacer)
            idx = mm_pos - 1
            flip = {"A": "G", "G": "A", "T": "C", "C": "T"}
            wt_seq[idx] = flip.get(wt_seq[idx], wt_seq[idx])
            wt_seq = "".join(wt_seq)

            report = generate_enhanced_variants(
                candidate_id=cid,
                target_label=cand.get("target_label", target_label),
                spacer_seq=mut_seq,
                wt_target_seq=wt_seq,
                mut_target_seq=mut_seq,
                natural_mm_position=mm_pos,
                config=config,
            )
            reports.append(report)
            if report.enhancement_possible:
                n_enhanced += 1

        all_reports[target_label] = reports

        if n_direct > 0:
            log.info(
                f"  {target_label}: {n_direct} direct candidates, "
                f"{n_enhanced} enhanced"
            )

    # Save enhancement results
    enhancement_path = output_dir / "enhancement_results.json"
    serialisable = {}
    for label, reports in all_reports.items():
        serialisable[label] = []
        for r in reports:
            entry = {
                "candidate_id": r.candidate_id,
                "target_label": r.target_label,
                "n_variants_generated": r.n_variants_generated,
                "n_variants_viable": r.n_variants_viable,
                "enhancement_possible": r.enhancement_possible,
                "natural_discrimination": r.natural_discrimination_score,
                "best_discrimination": r.best_discrimination_score,
                "improvement_factor": r.improvement_factor,
            }
            if r.best_variant:
                bv = r.best_variant
                entry["best_variant"] = {
                    "variant_id": bv.variant_id,
                    "original_spacer": bv.original_spacer_seq,
                    "enhanced_spacer": bv.enhanced_spacer_seq,
                    "synthetic_position": bv.synthetic_mismatches[0].position,
                    "activity_vs_mut": bv.predicted_activity_vs_mut,
                    "activity_vs_wt": bv.predicted_activity_vs_wt,
                    "discrimination_score": bv.discrimination_score,
                    "enhancement_type": bv.enhancement_type,
                }
            serialisable[label].append(entry)

    with open(enhancement_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    log.info(f"  Enhancement results saved: {enhancement_path}")

    return all_reports


# ======================================================================
# STEP 3: Publication figures
# ======================================================================

def step3_figures(output_dir: Path):
    """Generate all 11 publication-quality figures."""
    log.info("\n" + "=" * 70)
    log.info("  STEP 3: GENERATING PUBLICATION FIGURES (1–11)")
    log.info("=" * 70)

    try:
        from scripts.saber_pub_figures import generate_all_figures
    except ImportError:
        # Fallback: try direct import if running from scripts/
        try:
            from saber_pub_figures import generate_all_figures
        except ImportError:
            log.warning(
                "Cannot import saber_pub_figures. "
                "Make sure saber_pub_figures.py is in scripts/. "
                "Skipping figure generation."
            )
            return

    fig_dir = output_dir / "figures"
    generate_all_figures(fig_dir)
    log.info(f"  All figures saved to {fig_dir}/")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full MDR-TB 14-plex SABER pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-r", "--reference", required=True, help="H37Rv FASTA")
    parser.add_argument("-g", "--gff", required=True, help="H37Rv GFF3")
    parser.add_argument("-o", "--output", default="results/mdr_14plex_full",
                        help="Output directory")
    parser.add_argument("--skip-design", action="store_true",
                        help="Skip Step 1 if results already exist")
    parser.add_argument("--skip-enhance", action="store_true",
                        help="Skip Step 2 enhancement")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Skip Step 3 figure generation")
    parser.add_argument("--figures-only", action="store_true",
                        help="Only run Step 3 (figures)")
    args = parser.parse_args()

    out = Path(args.output)
    t_total = time.time()

    # ── Step 1: Panel Design ──
    if args.figures_only:
        log.info("--figures-only: skipping Steps 1-2")
    elif args.skip_design and (out / "panel_summary.json").exists():
        log.info("Skipping Step 1 (--skip-design, results exist)")
    else:
        step1_design(args.reference, args.gff, out)

    # ── Step 2: Synthetic Mismatch Enhancement ──
    if not args.figures_only and not args.skip_enhance:
        step2_enhance(out)
    elif args.skip_enhance:
        log.info("Skipping Step 2 (--skip-enhance)")

    # ── Step 3: Generate Figures ──
    if not args.skip_figures:
        step3_figures(out)
    else:
        log.info("Skipping Step 3 (--skip-figures)")

    elapsed = time.time() - t_total
    log.info(f"\n{'='*70}")
    log.info(f"  PIPELINE COMPLETE in {elapsed:.1f}s")
    log.info(f"  Output: {out}/")
    log.info(f"{'='*70}")


if __name__ == "__main__":
    main()
