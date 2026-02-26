#!/usr/bin/env python3
"""Full MDR-TB 14-plex pipeline: design → enhance → visualise.

Runs the complete SABER workflow:
  1. Design crRNA candidates for all 14 WHO-critical resistance mutations
  2. Apply synthetic mismatch enhancement on all direct candidates
  3. Generate publication-quality figures

Usage:
    python scripts/run_full_pipeline.py \
        -r data/references/H37Rv.fasta \
        -g data/references/H37Rv.gff3 \
        -o results/mdr_14plex_full
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Ensure saber is importable
_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("full_pipeline")

# ======================================================================
# STEP 1: Import panel design
# ======================================================================
from scripts.design_core_panel import PANEL_MUTATIONS, parse_mutation, run_panel

# ======================================================================
# STEP 2: Enhancement imports
# ======================================================================
from saber.candidates.synthetic_mismatch import (
    generate_enhanced_variants,
    EnhancementConfig,
    EnhancementReport,
)

# ======================================================================
# STEP 3: Visualization imports
# ======================================================================
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for file output
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    log.warning("matplotlib not installed — skipping figure generation")


# ======================================================================
# Enhancement runner
# ======================================================================

def run_enhancement(output_dir: Path, config: EnhancementConfig) -> dict:
    """Run synthetic mismatch enhancement on all scored candidates.

    Returns dict: target_label -> list of EnhancementReport
    """
    log.info("\n" + "=" * 70)
    log.info("  STEP 2: SYNTHETIC MISMATCH ENHANCEMENT")
    log.info("=" * 70)

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
# Figure generation
# ======================================================================

def generate_figures(output_dir: Path, enhancement_reports: dict) -> None:
    """Generate all publication-quality figures."""
    if not HAS_MPL:
        log.warning("Skipping figures — matplotlib not available")
        return

    log.info("\n" + "=" * 70)
    log.info("  STEP 3: GENERATING FIGURES")
    log.info("=" * 70)

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Set publication style
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Load panel summary
    summary_path = output_dir / "panel_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            panel_summary = json.load(f)
    else:
        panel_summary = []

    # Load all scored candidates per target
    all_scored = {}
    for d in sorted(output_dir.iterdir()):
        if not d.is_dir():
            continue
        sp = d / "scored_candidates.json"
        if sp.exists():
            with open(sp) as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                candidates = []
                for v in raw.values():
                    if isinstance(v, list):
                        candidates.extend(v)
                    else:
                        candidates.append(v)
            else:
                candidates = raw
            if candidates:
                all_scored[d.name] = candidates

    # ── Figure 1: Panel Overview ──
    _fig_panel_overview(panel_summary, fig_dir)

    # ── Figure 2: Candidate Ranking per Target ──
    _fig_candidate_ranking(all_scored, fig_dir)

    # ── Figure 3: Enhancement Discrimination Comparison ──
    _fig_enhancement_comparison(enhancement_reports, fig_dir)

    # ── Figure 4: PAM Landscape ──
    _fig_pam_landscape(all_scored, fig_dir)

    # ── Figure 5: Mismatch Position Heatmap ──
    _fig_mismatch_heatmap(enhancement_reports, fig_dir)

    log.info(f"  All figures saved to {fig_dir}/")


def _fig_panel_overview(panel_summary: list, fig_dir: Path) -> None:
    """Figure 1: Panel overview — candidates per target, colored by strategy."""
    if not panel_summary:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    genes = [f"{r['gene']} {r['mutation']}" for r in panel_summary]
    direct = [r.get("n_direct", 0) for r in panel_summary]
    prox = [r.get("n_proximity", 0) for r in panel_summary]
    drugs = [r.get("drug", "") for r in panel_summary]

    y = np.arange(len(genes))
    bar_height = 0.6

    # Drug color mapping
    drug_colors = {
        "RIF": "#2C6FAC", "INH": "#E74C3C", "EMB": "#27AE60",
        "PZA": "#F39C12", "LFX": "#8E44AD", "AMK": "#16A085",
        "KAN": "#D35400",
    }

    # Direct bars
    bars_d = ax.barh(y, direct, bar_height, label="Direct (mutation in seed)",
                     color="#2C6FAC", alpha=0.85, edgecolor="white", linewidth=0.5)
    # Proximity bars (stacked)
    bars_p = ax.barh(y, prox, bar_height, left=direct,
                     label="Proximity (AS-RPA discrimination)",
                     color="#E67E22", alpha=0.75, edgecolor="white", linewidth=0.5)

    # Drug badges on right
    for i, drug in enumerate(drugs):
        color = drug_colors.get(drug, "#95A5A6")
        total = direct[i] + prox[i]
        ax.text(max(direct[i] + prox[i] + 1, 22), i, drug,
                ha="left", va="center", fontsize=7, fontweight="bold",
                color="white", bbox=dict(boxstyle="round,pad=0.2",
                                         facecolor=color, edgecolor="none"))

    ax.set_yticks(y)
    ax.set_yticklabels(genes, fontsize=8)
    ax.set_xlabel("Number of crRNA candidates")
    ax.set_title("MDR-TB 14-Plex Panel: crRNA Candidate Inventory", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.invert_yaxis()

    # Mark failed targets
    for i, (d, p) in enumerate(zip(direct, prox)):
        if d + p == 0:
            ax.text(1, i, "NO CANDIDATES", ha="left", va="center",
                    fontsize=7, color="#E74C3C", fontstyle="italic")

    plt.tight_layout()
    path = fig_dir / "01_panel_overview.png"
    fig.savefig(path)
    plt.close(fig)
    log.info(f"  Figure 1: {path}")


def _fig_candidate_ranking(all_scored: dict, fig_dir: Path) -> None:
    """Figure 2: Top candidates per target ranked by heuristic score."""
    targets_with_data = {k: v for k, v in all_scored.items() if v}
    if not targets_with_data:
        return

    n_targets = len(targets_with_data)
    fig, axes = plt.subplots(
        min(n_targets, 7), 2,
        figsize=(14, min(n_targets, 7) * 2.2),
        squeeze=False,
    )

    target_list = sorted(targets_with_data.keys())

    for idx, target in enumerate(target_list[:14]):
        row = idx // 2
        col = idx % 2
        if row >= axes.shape[0]:
            break
        ax = axes[row, col]

        candidates = targets_with_data[target]
        # Extract scores
        scores = []
        labels = []
        strategies = []
        for c in candidates[:10]:
            cand = c.get("candidate", c)
            h = c.get("heuristic", {})
            score = h.get("composite", 0) if isinstance(h, dict) else 0
            spacer = cand.get("spacer_seq", "")[:15] + "..."
            strategy = str(cand.get("detection_strategy", "direct"))
            scores.append(score)
            labels.append(spacer)
            strategies.append(strategy)

        colors = ["#2C6FAC" if "DIRECT" in s.upper() or s == "direct"
                  else "#E67E22" for s in strategies]

        y = np.arange(len(scores))
        ax.barh(y, scores, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6, fontfamily="monospace")
        ax.set_xlim(0, 1)
        ax.set_title(target, fontsize=9, fontweight="bold")
        ax.invert_yaxis()

    # Hide empty subplots
    for idx in range(len(target_list), axes.shape[0] * 2):
        row, col = idx // 2, idx % 2
        if row < axes.shape[0]:
            axes[row, col].set_visible(False)

    fig.suptitle("crRNA Candidate Rankings by Heuristic Score", fontweight="bold", y=1.01)
    plt.tight_layout()
    path = fig_dir / "02_candidate_ranking.png"
    fig.savefig(path)
    plt.close(fig)
    log.info(f"  Figure 2: {path}")


def _fig_enhancement_comparison(reports: dict, fig_dir: Path) -> None:
    """Figure 3: Discrimination improvement from synthetic mismatches."""
    # Collect data from targets with direct candidates
    targets = []
    baselines = []
    enhanced = []

    for label, report_list in sorted(reports.items()):
        if not report_list:
            continue
        # Take best report for this target
        viable = [r for r in report_list if r.n_variants_viable > 0]
        if not viable:
            continue
        best = max(viable, key=lambda r: r.best_discrimination_score)
        targets.append(label.replace("_", " "))
        baselines.append(best.natural_discrimination_score)
        enhanced.append(best.best_discrimination_score)

    if not targets:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    y = np.arange(len(targets))
    bar_height = 0.35

    ax.barh(y - bar_height / 2, baselines, bar_height,
            label="Baseline (1 natural MM)", color="#95A5A6", alpha=0.85)
    ax.barh(y + bar_height / 2, enhanced, bar_height,
            label="Enhanced (+ synthetic MM)", color="#2C6FAC", alpha=0.85)

    # Improvement labels
    for i in range(len(targets)):
        if enhanced[i] > baselines[i] * 1.2:
            improvement = enhanced[i] / max(baselines[i], 0.1)
            ax.text(enhanced[i] + 0.3, y[i] + bar_height / 2,
                    f"{improvement:.1f}×", fontsize=7, va="center",
                    color="#2C6FAC", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(targets, fontsize=8)
    ax.set_xlabel("Predicted Discrimination Ratio (MUT/WT activity)")
    ax.set_title("Synthetic Mismatch Enhancement: Discrimination Improvement",
                 fontweight="bold")
    ax.legend(loc="lower right")
    ax.axvline(x=10, color="#E74C3C", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.text(10.5, len(targets) - 0.5, "diagnostic\nthreshold",
            fontsize=7, color="#E74C3C", alpha=0.7)

    plt.tight_layout()
    path = fig_dir / "03_enhancement_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    log.info(f"  Figure 3: {path}")


def _fig_pam_landscape(all_scored: dict, fig_dir: Path) -> None:
    """Figure 4: PAM variant distribution across targets."""
    pam_counts = {}  # target -> {pam_variant: count}
    for target, candidates in sorted(all_scored.items()):
        pam_counts[target] = {}
        for c in candidates:
            cand = c.get("candidate", c)
            pam = cand.get("pam_variant", "unknown")
            # Clean enum string
            if "." in str(pam):
                pam = str(pam).split(".")[-1]
            pam_counts[target][pam] = pam_counts[target].get(pam, 0) + 1

    if not pam_counts:
        return

    # Get all PAM types
    all_pams = sorted(set(p for counts in pam_counts.values() for p in counts))
    targets = sorted(pam_counts.keys())

    # Build matrix
    matrix = np.zeros((len(targets), len(all_pams)))
    for i, t in enumerate(targets):
        for j, p in enumerate(all_pams):
            matrix[i, j] = pam_counts[t].get(p, 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = LinearSegmentedColormap.from_list("saber", ["#F8F9FA", "#2C6FAC"])
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")

    ax.set_xticks(np.arange(len(all_pams)))
    ax.set_xticklabels(all_pams, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(targets)))
    ax.set_yticklabels([t.replace("_", " ") for t in targets], fontsize=8)

    # Annotate cells
    for i in range(len(targets)):
        for j in range(len(all_pams)):
            val = int(matrix[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=7, color="white" if val > 5 else "black")

    ax.set_title("PAM Variant Distribution Across Targets", fontweight="bold")
    plt.colorbar(im, ax=ax, label="Candidate count", shrink=0.8)
    plt.tight_layout()
    path = fig_dir / "04_pam_landscape.png"
    fig.savefig(path)
    plt.close(fig)
    log.info(f"  Figure 4: {path}")


def _fig_mismatch_heatmap(reports: dict, fig_dir: Path) -> None:
    """Figure 5: Synthetic mismatch position vs discrimination heatmap."""
    # Collect all enhancement variants across targets
    positions = []
    discriminations = []
    activities = []
    target_labels = []

    for label, report_list in sorted(reports.items()):
        for report in report_list:
            for variant in report.all_variants[:5]:
                for s in variant.synthetic_mismatches:
                    positions.append(s.position)
                    discriminations.append(variant.discrimination_score)
                    activities.append(variant.predicted_activity_vs_mut)
                    target_labels.append(label)

    if not positions:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Position vs Discrimination scatter
    scatter = ax1.scatter(positions, discriminations,
                          c=activities, cmap="RdYlGn", s=40, alpha=0.7,
                          edgecolors="white", linewidth=0.5, vmin=0, vmax=1)
    ax1.set_xlabel("Synthetic Mismatch Position (from PAM)")
    ax1.set_ylabel("Predicted Discrimination Ratio")
    ax1.set_title("a) Position vs Discrimination", fontweight="bold")
    ax1.axhline(y=10, color="#E74C3C", linestyle="--", alpha=0.5)
    ax1.text(max(positions) + 0.3, 10, "10× threshold", fontsize=7,
             color="#E74C3C", va="bottom")
    plt.colorbar(scatter, ax=ax1, label="Activity vs MUT", shrink=0.8)

    # Right: Position histogram colored by whether enhancement worked
    pos_enhanced = [p for p, d in zip(positions, discriminations) if d > 5]
    pos_weak = [p for p, d in zip(positions, discriminations) if d <= 5]

    bins = np.arange(0.5, max(positions) + 1.5, 1)
    ax2.hist([pos_enhanced, pos_weak], bins=bins, stacked=True,
             color=["#2C6FAC", "#95A5A6"],
             label=["Disc > 5×", "Disc ≤ 5×"],
             alpha=0.85, edgecolor="white")
    ax2.set_xlabel("Synthetic Mismatch Position (from PAM)")
    ax2.set_ylabel("Number of Variants")
    ax2.set_title("b) Effective Enhancement by Position", fontweight="bold")
    ax2.legend()

    # Seed region annotation
    ax2.axvspan(0.5, 8.5, alpha=0.08, color="#2C6FAC")
    ax2.text(4.5, ax2.get_ylim()[1] * 0.9, "SEED", ha="center",
             fontsize=8, color="#2C6FAC", alpha=0.6, fontweight="bold")

    plt.tight_layout()
    path = fig_dir / "05_mismatch_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    log.info(f"  Figure 5: {path}")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Full MDR-TB SABER pipeline")
    parser.add_argument("-r", "--reference", required=True, help="H37Rv FASTA")
    parser.add_argument("-g", "--gff", required=True, help="H37Rv GFF3")
    parser.add_argument("-o", "--output", default="results/mdr_14plex_full",
                        help="Output directory")
    parser.add_argument("--skip-design", action="store_true",
                        help="Skip Step 1 if results already exist")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Skip Step 3 figure generation")
    args = parser.parse_args()

    out = Path(args.output)
    t_total = time.time()

    # ── Step 1: Panel Design ──
    if args.skip_design and (out / "panel_summary.json").exists():
        log.info("Skipping Step 1 (--skip-design, results exist)")
    else:
        log.info("\n" + "=" * 70)
        log.info("  STEP 1: DESIGNING 14-PLEX MDR-TB PANEL")
        log.info("=" * 70)
        run_panel(
            reference=args.reference,
            gff=args.gff,
            output_dir=str(out),
        )

    # ── Step 2: Synthetic Mismatch Enhancement ──
    enhancement_config = EnhancementConfig(
        cas_variant="enAsCas12a",
        allow_double_synthetic=True,
        min_activity_vs_mut=0.25,
        search_radius=6,
    )
    enhancement_reports = run_enhancement(out, enhancement_config)

    # ── Step 3: Generate Figures ──
    if not args.skip_figures:
        generate_figures(out, enhancement_reports)

    elapsed = time.time() - t_total
    log.info(f"\n{'='*70}")
    log.info(f"  PIPELINE COMPLETE in {elapsed:.1f}s")
    log.info(f"  Output: {out}/")
    log.info(f"{'='*70}")


if __name__ == "__main__":
    main()
