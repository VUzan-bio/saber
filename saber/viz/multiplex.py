"""Multiplex compatibility visualisation.

14×14 cross-reactivity matrix and 28×28 primer dimer matrix.
Essential for multiplex panel design — identifies conflicts
that require alternative candidate selection.

Inspired by:
- Multiplex PCR primer compatibility heatmaps
- Cas12a cross-reactivity panels in diagnostic papers
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from saber.viz.style import (
    CMAP_SEQ, PALETTE, apply_style, add_panel_label,
    DOUBLE_COL, save_figure,
)


class MultiplexMatrixPlot:
    """Visualise multiplex panel compatibility.

    Usage:
        plot = MultiplexMatrixPlot()
        fig = plot.plot_cross_reactivity(matrix, labels)
        fig = plot.plot_primer_dimers(dg_matrix, primer_labels)
        fig = plot.plot_combined(cross_matrix, dimer_matrix, labels)
    """

    def plot_cross_reactivity(
        self,
        matrix: np.ndarray,
        labels: list[str],
        threshold: float = 0.3,
    ) -> plt.Figure:
        """N×N crRNA cross-reactivity matrix.

        Values = sequence similarity or predicted cross-activation.
        Cells above threshold are flagged.

        Args:
            matrix: (N, N) symmetric matrix, values 0-1.
            labels: Target labels (e.g. ["rpoB_S531L", "katG_S315T", ...]).
            threshold: Flag cells above this value.
        """
        apply_style()

        n = matrix.shape[0]
        fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.6, DOUBLE_COL * 0.55))

        # Mask diagonal
        mask = np.eye(n, dtype=bool)
        display = np.ma.masked_where(mask, matrix)

        im = ax.imshow(display, cmap=CMAP_SEQ, vmin=0, vmax=0.6, interpolation="nearest")

        # Flag high cross-reactivity
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                val = matrix[i, j]
                if val > threshold:
                    rect = patches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        linewidth=1.5, edgecolor=PALETTE["red"],
                        facecolor="none",
                    )
                    ax.add_patch(rect)
                # Value annotation
                color = "white" if val > 0.35 else PALETTE["dark"]
                if not mask[i, j]:
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=4, color=color)

        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=5)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=5)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Sequence similarity", fontsize=6)
        cbar.ax.tick_params(labelsize=5)

        ax.set_title("crRNA cross-reactivity matrix", fontsize=8, fontweight="bold")

        fig.tight_layout()
        return fig

    def plot_primer_dimers(
        self,
        dg_matrix: np.ndarray,
        labels: list[str],
        threshold: float = -6.0,
    ) -> plt.Figure:
        """2N×2N primer dimer ΔG matrix for the multiplex panel.

        Args:
            dg_matrix: (2N, 2N) matrix of dimer ΔG values (kcal/mol).
                       More negative = stronger dimer = worse.
            labels: Primer labels (e.g. ["rpoB_fwd", "rpoB_rev", ...]).
            threshold: ΔG below this is flagged as problematic.
        """
        apply_style()

        n = dg_matrix.shape[0]

        # Colourmap: white (safe) → red (dangerous dimers)
        import matplotlib.colors as mcolors
        cmap_dimer = mcolors.LinearSegmentedColormap.from_list(
            "dimer", ["#FFFFFF", "#FADBD8", "#E74C3C", "#922B21"]
        )

        fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.7, DOUBLE_COL * 0.65))

        # Flip sign for display (more negative = more red)
        display = np.abs(dg_matrix)
        display[display == 0] = np.nan

        im = ax.imshow(display, cmap=cmap_dimer, vmin=0, vmax=10, interpolation="nearest")

        # Flag dangerous dimers
        conflicts = 0
        for i in range(n):
            for j in range(i + 1, n):
                if dg_matrix[i, j] < threshold:
                    rect = patches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        linewidth=1.5, edgecolor=PALETTE["red"],
                        facecolor="none",
                    )
                    ax.add_patch(rect)
                    conflicts += 1

        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=3.5)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=3.5)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("|ΔG| (kcal/mol)", fontsize=6)
        cbar.ax.tick_params(labelsize=5)

        ax.set_title(
            f"Primer dimer matrix — {conflicts} conflicts (ΔG < {threshold} kcal/mol)",
            fontsize=7, fontweight="bold",
        )

        fig.tight_layout()
        return fig

    def plot_combined(
        self,
        cross_matrix: np.ndarray,
        dimer_matrix: np.ndarray,
        target_labels: list[str],
    ) -> plt.Figure:
        """Combined figure: cross-reactivity (left) + primer dimers (right).

        The standard multiplex compatibility figure for the paper/thesis.
        """
        apply_style()

        n_targets = cross_matrix.shape[0]
        n_primers = dimer_matrix.shape[0]

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.45),
            gridspec_kw={"width_ratios": [1, 1.2], "wspace": 0.35},
        )

        # Left: cross-reactivity
        mask = np.eye(n_targets, dtype=bool)
        display = np.ma.masked_where(mask, cross_matrix)
        im1 = ax1.imshow(display, cmap=CMAP_SEQ, vmin=0, vmax=0.6)
        ax1.set_xticks(range(n_targets))
        ax1.set_xticklabels(target_labels, rotation=45, ha="right", fontsize=4)
        ax1.set_yticks(range(n_targets))
        ax1.set_yticklabels(target_labels, fontsize=4)
        ax1.set_title("crRNA cross-reactivity", fontsize=7, fontweight="bold")
        fig.colorbar(im1, ax=ax1, shrink=0.7, pad=0.02).ax.tick_params(labelsize=4)
        add_panel_label(ax1, "a")

        # Right: primer dimers
        import matplotlib.colors as mcolors
        cmap_d = mcolors.LinearSegmentedColormap.from_list(
            "dimer2", ["#FFFFFF", "#FADBD8", "#E74C3C"]
        )
        display2 = np.abs(dimer_matrix)
        display2[display2 == 0] = np.nan
        im2 = ax2.imshow(display2, cmap=cmap_d, vmin=0, vmax=10)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title("Primer dimer ΔG", fontsize=7, fontweight="bold")
        fig.colorbar(im2, ax=ax2, shrink=0.7, pad=0.02).ax.tick_params(labelsize=4)
        add_panel_label(ax2, "b")

        fig.tight_layout()
        return fig
