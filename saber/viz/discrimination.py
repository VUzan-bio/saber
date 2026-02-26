"""Discrimination heatmap — mismatch tolerance landscape.

The key figure for CRISPR-Cas12a SNV diagnostics. Shows predicted or measured
trans-cleavage activity across all spacer positions (1-20) and mismatch types
(3 possible substitutions per position = 60 cells).

Directly inspired by:
- ARTEMIS (Kohabir et al., Cell Reports Methods 2024) seed region heatmaps
- Iterative crRNA design (Comm. Biology 2024) mismatch profiling panels
- Kim et al. (Nature Biotech 2018) Cas12a activity matrices

The heatmap reveals WHERE a guide is tolerant vs intolerant of mismatches,
which is the fundamental information for single-nucleotide discrimination.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from saber.viz.style import (
    CMAP_DIV, PALETTE, apply_style, add_panel_label,
    DOUBLE_COL, save_figure,
)


class DiscriminationHeatmap:
    """Generate mismatch discrimination heatmaps.

    Usage:
        hmap = DiscriminationHeatmap()

        # From mismatch pair predictions
        fig = hmap.plot(activity_matrix, spacer_seq, target_label)

        # Side-by-side WT-detecting vs MUT-detecting crRNAs
        fig = hmap.plot_comparison(matrix_wt, matrix_mut, spacer_wt, spacer_mut, label)

        # Multi-target panel overview
        fig = hmap.plot_panel_summary(matrices_dict)
    """

    def plot(
        self,
        activity_matrix: np.ndarray,
        spacer_seq: str,
        target_label: str,
        mutation_pos: Optional[int] = None,
        vmin: float = 0.0,
        vmax: float = 1.0,
        figsize: Optional[tuple[float, float]] = None,
    ) -> plt.Figure:
        """Single crRNA discrimination heatmap.

        Args:
            activity_matrix: (3, L) array where rows = substitution types (A→X, T→X, etc.)
                             and columns = spacer positions 1..L. Values = predicted/measured
                             normalised activity (0 = no cleavage, 1 = full activity).
            spacer_seq: The spacer nucleotide sequence.
            target_label: e.g. "rpoB_S531L".
            mutation_pos: 1-indexed position of the resistance mutation in the spacer.
                          Highlighted with a box.
            vmin, vmax: Colour scale limits.
        """
        apply_style()

        L = activity_matrix.shape[1]
        if figsize is None:
            figsize = (DOUBLE_COL, DOUBLE_COL * 0.35)

        fig, ax = plt.subplots(figsize=figsize)

        # Substitution labels for each row
        ref_nts = list(spacer_seq.upper())
        sub_labels = self._get_substitution_labels(ref_nts)

        im = ax.imshow(
            activity_matrix, aspect="auto", cmap=CMAP_DIV,
            vmin=vmin, vmax=vmax, interpolation="nearest",
        )

        # Axes
        ax.set_xticks(range(L))
        ax.set_xticklabels(
            [f"{i+1}\n{ref_nts[i]}" for i in range(L)],
            fontsize=5, ha="center",
        )
        ax.set_yticks(range(3))
        ax.set_yticklabels(["Sub 1", "Sub 2", "Sub 3"], fontsize=6)

        # Proper substitution labels on y-axis
        for row in range(3):
            for col in range(L):
                val = activity_matrix[row, col]
                color = "white" if val > 0.6 else PALETTE["dark"]
                ax.text(
                    col, row, f"{val:.2f}", ha="center", va="center",
                    fontsize=4.5, color=color, fontweight="medium",
                )

        # Highlight seed region (positions 1-8)
        seed_rect = patches.FancyBboxPatch(
            (-0.5, -0.5), 8, 3,
            boxstyle="round,pad=0.05",
            linewidth=1.5, edgecolor=PALETTE["blue"],
            facecolor="none", linestyle="--",
        )
        ax.add_patch(seed_rect)
        ax.text(3.5, -0.9, "Seed region", ha="center", fontsize=5,
                color=PALETTE["blue"], fontweight="bold")

        # Highlight mutation position
        if mutation_pos is not None:
            mut_rect = patches.Rectangle(
                (mutation_pos - 1.5, -0.5), 1, 3,
                linewidth=2, edgecolor=PALETTE["red"],
                facecolor="none",
            )
            ax.add_patch(mut_rect)

        # Colour bar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Normalised activity", fontsize=6)
        cbar.ax.tick_params(labelsize=5)

        # Labels
        ax.set_xlabel("Spacer position (from PAM)", fontsize=7, fontweight="bold")
        ax.set_title(
            f"Mismatch discrimination landscape — {target_label}",
            fontsize=8, fontweight="bold", pad=12,
        )

        # PAM indicator
        ax.annotate(
            "PAM →", xy=(-0.5, 1.5), fontsize=6, color=PALETTE["grey"],
            ha="right", va="center",
        )

        fig.tight_layout()
        return fig

    def plot_comparison(
        self,
        matrix_wt: np.ndarray,
        matrix_mut: np.ndarray,
        spacer_wt: str,
        spacer_mut: str,
        target_label: str,
        mutation_pos: int = 4,
    ) -> plt.Figure:
        """Side-by-side comparison of WT-detecting vs MUT-detecting crRNAs.

        Standard figure in CRISPR diagnostic papers: shows that MUT-targeting
        crRNA has low WT activity (= good discrimination) and vice versa.
        """
        apply_style()

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.35),
            sharey=True, gridspec_kw={"wspace": 0.08},
        )

        for ax, matrix, spacer, title, label in [
            (ax1, matrix_mut, spacer_mut, "MUT-targeting crRNA", "a"),
            (ax2, matrix_wt, spacer_wt, "WT-targeting crRNA", "b"),
        ]:
            L = matrix.shape[1]
            ref_nts = list(spacer.upper())

            im = ax.imshow(
                matrix, aspect="auto", cmap=CMAP_DIV,
                vmin=0, vmax=1, interpolation="nearest",
            )
            ax.set_xticks(range(L))
            ax.set_xticklabels(
                [f"{i+1}\n{ref_nts[i]}" for i in range(L)], fontsize=4.5,
            )
            ax.set_title(title, fontsize=7, fontweight="bold")

            # Seed region
            seed = patches.FancyBboxPatch(
                (-0.5, -0.5), 8, 3,
                boxstyle="round,pad=0.05",
                linewidth=1.2, edgecolor=PALETTE["blue"],
                facecolor="none", linestyle="--",
            )
            ax.add_patch(seed)

            # Mutation position
            mut = patches.Rectangle(
                (mutation_pos - 1.5, -0.5), 1, 3,
                linewidth=1.5, edgecolor=PALETTE["red"], facecolor="none",
            )
            ax.add_patch(mut)

            add_panel_label(ax, label)

        ax1.set_ylabel("Substitution", fontsize=6)
        ax1.set_yticks(range(3))
        ax1.set_yticklabels(["Sub 1", "Sub 2", "Sub 3"], fontsize=5)

        fig.suptitle(
            f"Discrimination comparison — {target_label}",
            fontsize=8, fontweight="bold", y=1.02,
        )

        cbar = fig.colorbar(im, ax=[ax1, ax2], shrink=0.8, pad=0.02)
        cbar.set_label("Normalised activity", fontsize=6)
        cbar.ax.tick_params(labelsize=5)

        fig.tight_layout()
        return fig

    def plot_panel_summary(
        self,
        discrimination_ratios: dict[str, float],
        threshold: float = 2.0,
    ) -> plt.Figure:
        """Bar chart summary of discrimination ratios across all panel targets.

        Quick overview: which targets have good discrimination (ratio >> 1)
        and which need alternative candidates.
        """
        apply_style()

        labels = list(discrimination_ratios.keys())
        ratios = list(discrimination_ratios.values())
        n = len(labels)

        colors = [
            PALETTE["green"] if r >= threshold else
            PALETTE["orange"] if r >= 1.0 else
            PALETTE["red"]
            for r in ratios
        ]

        fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.3))

        bars = ax.barh(range(n), ratios, color=colors, edgecolor="white", height=0.7)
        ax.axvline(threshold, color=PALETTE["dark"], linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(threshold + 0.1, n - 0.5, f"Threshold ({threshold}×)",
                fontsize=5, color=PALETTE["grey"])

        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("Discrimination ratio (MUT / WT activity)", fontsize=7)
        ax.set_title("Panel discrimination overview", fontsize=8, fontweight="bold")
        ax.invert_yaxis()

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _get_substitution_labels(ref_nts: list[str]) -> list[list[str]]:
        """For each position, return the 3 possible substitutions."""
        bases = ["A", "T", "G", "C"]
        labels = []
        for nt in ref_nts:
            subs = [b for b in bases if b != nt.upper()]
            labels.append(subs)
        return labels
