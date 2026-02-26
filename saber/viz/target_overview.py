"""Target dashboard — PhD project management figure.

Overview of all 14 targets with gene, mutation, drug, candidate count,
top score, and validation status. Colour-coded by readiness.

This is the "at a glance" figure for supervisors, collaborators, and
progress reports.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from saber.viz.style import (
    PALETTE, apply_style, add_panel_label,
    DOUBLE_COL, save_figure,
)


class TargetDashboard:
    """Project overview visualisation.

    Usage:
        dash = TargetDashboard()
        fig = dash.plot_dashboard(targets_data)
        fig = dash.plot_gene_locus_map(gene_data)
    """

    STATUS_COLORS = {
        "untested": "#DAEAF6",
        "in_progress": "#FDEBD0",
        "validated": "#D5F5E3",
        "failed": "#FADBD8",
    }

    STATUS_BORDER = {
        "untested": "#AED6F1",
        "in_progress": "#F5CBA7",
        "validated": "#82E0AA",
        "failed": "#F1948A",
    }

    DRUG_COLORS = {
        "RIF": "#2C6FAC",
        "INH": "#27AE60",
        "EMB": "#E67E22",
        "PZA": "#8E44AD",
        "FQ": "#C0392B",
        "AG": "#16A085",
        "BDQ": "#D4AC0D",
        "LZD": "#6C3483",
    }

    def plot_dashboard(
        self,
        targets: list[dict],
    ) -> plt.Figure:
        """Grid-based target dashboard.

        Args:
            targets: List of dicts, each with:
                - gene: str
                - mutation: str (e.g. "S531L")
                - drug: str (e.g. "RIF")
                - n_candidates: int
                - top_score: float
                - status: str ("untested"|"in_progress"|"validated"|"failed")
                - experimental_ratio: Optional[float]
        """
        apply_style()

        n = len(targets)
        cols = min(7, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(
            rows, cols,
            figsize=(DOUBLE_COL, rows * 1.1),
            gridspec_kw={"hspace": 0.4, "wspace": 0.25},
        )

        if rows == 1:
            axes = axes.reshape(1, -1)

        for idx, target in enumerate(targets):
            r, c = divmod(idx, cols)
            ax = axes[r, c]

            status = target.get("status", "untested")
            bg = self.STATUS_COLORS[status]
            border = self.STATUS_BORDER[status]
            drug = target.get("drug", "")
            drug_color = self.DRUG_COLORS.get(drug, PALETTE["grey"])

            # Card background
            card = patches.FancyBboxPatch(
                (0.02, 0.02), 0.96, 0.96,
                boxstyle="round,pad=0.05",
                facecolor=bg, edgecolor=border, linewidth=1.2,
                transform=ax.transAxes,
            )
            ax.add_patch(card)

            # Gene + mutation (title)
            ax.text(
                0.5, 0.85, f"{target['gene']}",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=7, fontweight="bold", color=PALETTE["dark"],
            )
            ax.text(
                0.5, 0.68, f"{target['mutation']}",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8, fontweight="bold", color=drug_color,
            )

            # Drug badge
            ax.text(
                0.5, 0.52, drug,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=5, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=drug_color,
                          edgecolor="none"),
            )

            # Stats
            n_cand = target.get("n_candidates", 0)
            top = target.get("top_score", 0)
            ax.text(
                0.5, 0.35,
                f"{n_cand} candidates",
                transform=ax.transAxes, ha="center", fontsize=5,
                color=PALETTE["dark"],
            )
            ax.text(
                0.5, 0.22,
                f"top: {top:.2f}",
                transform=ax.transAxes, ha="center", fontsize=5,
                color=PALETTE["dark"],
            )

            # Status indicator
            ax.text(
                0.5, 0.08, status.replace("_", " ").upper(),
                transform=ax.transAxes, ha="center", fontsize=4,
                color=PALETTE["grey"], fontweight="bold",
            )

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

        # Hide empty cells
        for idx in range(n, rows * cols):
            r, c = divmod(idx, cols)
            axes[r, c].axis("off")

        fig.suptitle(
            "SABER Target Dashboard — MDR-TB 14-plex Panel",
            fontsize=9, fontweight="bold", y=1.02,
        )

        return fig

    def plot_gene_locus_map(
        self,
        targets: list[dict],
        genome_length: int = 4_411_532,
    ) -> plt.Figure:
        """Linear genome map showing target positions on H37Rv.

        Circular chromosome linearised, with markers at each target locus.
        Colour-coded by drug resistance.
        """
        apply_style()

        fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.2))

        # Genome backbone
        ax.plot([0, genome_length], [0, 0], color=PALETTE["border"],
                linewidth=3, solid_capstyle="round")

        # Scale bar
        mb_ticks = range(0, genome_length + 1, 500_000)
        for pos in mb_ticks:
            ax.plot([pos, pos], [-0.15, 0.15], color=PALETTE["border"], linewidth=0.5)
            ax.text(pos, -0.35, f"{pos/1e6:.1f}", ha="center", fontsize=4,
                    color=PALETTE["grey"])
        ax.text(genome_length / 2, -0.6, "Position (Mb)", ha="center",
                fontsize=5, color=PALETTE["grey"])

        # Target markers
        for i, t in enumerate(targets):
            pos = t.get("genomic_pos", 0)
            drug = t.get("drug", "")
            color = self.DRUG_COLORS.get(drug, PALETTE["grey"])
            gene = t.get("gene", "")

            # Alternating label positions to avoid overlap
            y_offset = 0.5 if i % 2 == 0 else 1.0
            y_line = 0.15

            ax.plot(pos, 0, marker="v", color=color, markersize=5, zorder=5)
            ax.plot([pos, pos], [y_line, y_offset - 0.1], color=color,
                    linewidth=0.5, alpha=0.6)
            ax.text(pos, y_offset, f"{gene}\n{t.get('mutation', '')}",
                    ha="center", fontsize=4, color=color, fontweight="bold")

        ax.set_xlim(-100_000, genome_length + 100_000)
        ax.set_ylim(-0.8, 1.5)
        ax.axis("off")

        ax.set_title("Target loci on M. tuberculosis H37Rv genome",
                     fontsize=8, fontweight="bold")

        fig.tight_layout()
        return fig
