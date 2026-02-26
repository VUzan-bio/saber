"""Active learning cycle visualisation.

Tracks how JEPA prediction accuracy improves over experimental validation
cycles. This is the key figure demonstrating the active learning contribution:
each round of wet-lab data makes the next prediction round better.

Inspired by:
- Active learning convergence plots in ML literature
- Transfer learning fine-tuning curves (DeepCRISTL, Bioinformatics 2024)
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from saber.viz.style import (
    PALETTE, MODEL_COLORS, apply_style, add_panel_label,
    DOUBLE_COL, SINGLE_COL, save_figure,
)


class ActiveLearningPlot:
    """Visualise active learning cycle progress.

    Usage:
        plot = ActiveLearningPlot()
        fig = plot.plot_convergence(cycle_metrics)
        fig = plot.plot_residuals_by_cycle(cycles_data)
        fig = plot.plot_cycle_panel(cycle_data)
    """

    def plot_convergence(
        self,
        cycle_metrics: list[dict],
        metric_name: str = "spearman_rho",
    ) -> plt.Figure:
        """Line plot showing metric improvement over AL cycles.

        Args:
            cycle_metrics: List of dicts per cycle, e.g.:
                [{"cycle": 1, "spearman_rho": 0.35, "n_samples": 20,
                  "mae": 0.25, "model": "jepa_v1"},
                 {"cycle": 2, "spearman_rho": 0.52, "n_samples": 40, ...}]
            metric_name: Which metric to plot on y-axis.
        """
        apply_style()

        cycles = [d["cycle"] for d in cycle_metrics]
        values = [d[metric_name] for d in cycle_metrics]
        n_samples = [d.get("n_samples", 0) for d in cycle_metrics]

        fig, ax1 = plt.subplots(figsize=(SINGLE_COL * 1.3, SINGLE_COL * 0.9))

        # Primary: metric over cycles
        ax1.plot(cycles, values, "o-", color=PALETTE["blue"],
                 linewidth=1.5, markersize=6, markerfacecolor="white",
                 markeredgewidth=1.5, markeredgecolor=PALETTE["blue"])

        ax1.set_xlabel("Active learning cycle", fontsize=7)
        ax1.set_ylabel(self._metric_label(metric_name), fontsize=7,
                       color=PALETTE["blue"])
        ax1.tick_params(axis="y", labelcolor=PALETTE["blue"])
        ax1.set_xticks(cycles)

        # Heuristic baseline reference
        if len(cycle_metrics) > 0 and "heuristic_baseline" in cycle_metrics[0]:
            baseline = cycle_metrics[0]["heuristic_baseline"]
            ax1.axhline(baseline, color=PALETTE["grey"], linestyle="--",
                        linewidth=0.8, alpha=0.6)
            ax1.text(cycles[-1] + 0.1, baseline, "Heuristic\nbaseline",
                     fontsize=5, color=PALETTE["grey"], va="center")

        # Secondary: cumulative samples
        ax2 = ax1.twinx()
        ax2.bar(cycles, n_samples, alpha=0.15, color=PALETTE["teal"],
                width=0.5, edgecolor="none")
        ax2.set_ylabel("Cumulative samples", fontsize=6,
                       color=PALETTE["teal"], alpha=0.7)
        ax2.tick_params(axis="y", labelcolor=PALETTE["teal"], labelsize=5)

        ax1.set_title("Active learning convergence", fontsize=8, fontweight="bold")
        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)

        fig.tight_layout()
        return fig

    def plot_residuals_by_cycle(
        self,
        cycles_data: dict[int, tuple[np.ndarray, np.ndarray]],
    ) -> plt.Figure:
        """Residual distribution per cycle — shows error shrinking over time.

        Args:
            cycles_data: {cycle_number: (predicted, measured)} arrays.
        """
        apply_style()

        n_cycles = len(cycles_data)
        fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, SINGLE_COL * 0.7))

        positions = []
        data = []
        colors_list = []

        for i, (cycle, (pred, meas)) in enumerate(sorted(cycles_data.items())):
            residuals = pred - meas
            data.append(residuals)
            positions.append(i)

        bp = ax.boxplot(
            data, positions=positions, widths=0.5,
            patch_artist=True, showfliers=True,
            flierprops=dict(marker=".", markersize=2, alpha=0.3),
            medianprops=dict(color=PALETTE["red"], linewidth=1.2),
            whiskerprops=dict(color=PALETTE["dark"], linewidth=0.6),
            capprops=dict(color=PALETTE["dark"], linewidth=0.6),
        )

        # Colour gradient: lighter blue → darker blue over cycles
        blues = plt.cm.Blues(np.linspace(0.2, 0.8, n_cycles))
        for patch, color in zip(bp["boxes"], blues):
            patch.set_facecolor(color)
            patch.set_edgecolor(PALETTE["dark"])
            patch.set_linewidth(0.5)

        ax.axhline(0, color=PALETTE["grey"], linestyle="-", linewidth=0.5, alpha=0.4)

        ax.set_xticks(positions)
        ax.set_xticklabels([f"Cycle {c}" for c in sorted(cycles_data.keys())], fontsize=6)
        ax.set_ylabel("Prediction residual", fontsize=7)
        ax.set_title("Residual distribution by AL cycle", fontsize=8, fontweight="bold")

        fig.tight_layout()
        return fig

    def plot_cycle_panel(
        self,
        cycles_data: dict[int, tuple[np.ndarray, np.ndarray]],
    ) -> plt.Figure:
        """Multi-panel: scatter per cycle + convergence line.

        The definitive active learning figure for the thesis/paper.
        Top row: predicted vs measured scatter per cycle.
        Bottom: convergence metrics line plot.
        """
        apply_style()
        from scipy import stats

        n_cycles = len(cycles_data)
        sorted_cycles = sorted(cycles_data.keys())

        fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.55))

        # Top row: scatter per cycle
        gs = fig.add_gridspec(2, n_cycles, height_ratios=[1.5, 1], hspace=0.4, wspace=0.15)

        rhos = []
        maes = []
        panel_labels = "abcdefgh"

        for i, cycle in enumerate(sorted_cycles):
            pred, meas = cycles_data[cycle]
            rho, _ = stats.spearmanr(pred, meas)
            mae = np.mean(np.abs(pred - meas))
            rhos.append(rho)
            maes.append(mae)

            ax = fig.add_subplot(gs[0, i])
            color = MODEL_COLORS[i % len(MODEL_COLORS)]
            ax.scatter(meas, pred, c=color, s=6, alpha=0.4, edgecolors="none")
            ax.plot([0, 1], [0, 1], color=PALETTE["grey"], linewidth=0.5,
                    linestyle="--", alpha=0.4)

            ax.text(0.05, 0.90, f"ρ = {rho:.2f}\nn = {len(pred)}",
                    transform=ax.transAxes, fontsize=5,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=PALETTE["border"], alpha=0.9))

            ax.set_title(f"Cycle {cycle}", fontsize=7, fontweight="bold")
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect("equal")
            if i == 0:
                ax.set_ylabel("Predicted", fontsize=6)
            ax.set_xlabel("Measured", fontsize=5)
            add_panel_label(ax, panel_labels[i])

        # Bottom row: convergence
        ax_conv = fig.add_subplot(gs[1, :])
        ax_conv.plot(sorted_cycles, rhos, "o-", color=PALETTE["blue"],
                     linewidth=1.5, markersize=5, label="Spearman ρ")

        ax_conv2 = ax_conv.twinx()
        ax_conv2.plot(sorted_cycles, maes, "s--", color=PALETTE["red"],
                      linewidth=1, markersize=4, alpha=0.7, label="MAE")
        ax_conv2.set_ylabel("MAE", fontsize=6, color=PALETTE["red"])
        ax_conv2.tick_params(axis="y", labelcolor=PALETTE["red"], labelsize=5)

        ax_conv.set_xlabel("Active learning cycle", fontsize=7)
        ax_conv.set_ylabel("Spearman ρ", fontsize=6, color=PALETTE["blue"])
        ax_conv.tick_params(axis="y", labelcolor=PALETTE["blue"])
        ax_conv.set_xticks(sorted_cycles)
        ax_conv.set_title("Prediction accuracy convergence", fontsize=7, fontweight="bold")

        lines1, labels1 = ax_conv.get_legend_handles_labels()
        lines2, labels2 = ax_conv2.get_legend_handles_labels()
        ax_conv.legend(lines1 + lines2, labels1 + labels2, fontsize=5, loc="center right")

        add_panel_label(ax_conv, panel_labels[n_cycles])

        return fig

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _metric_label(name: str) -> str:
        return {
            "spearman_rho": "Spearman ρ",
            "mae": "Mean absolute error",
            "rmse": "RMSE",
            "r2": "R²",
        }.get(name, name)
