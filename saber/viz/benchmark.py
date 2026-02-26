"""Model benchmarking figures.

Compares heuristic, Seq-deepCpf1 baseline, and JEPA predictor performance.
Standard figures in the CRISPR guide prediction literature:
- Spearman correlation bar chart (like Konstantakos et al. 2022, NAR)
- Predicted vs measured scatter with regression (like CRISPRon, Kim 2018)
- ROC / precision-recall for binary guide classification
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from saber.viz.style import (
    PALETTE, MODEL_COLORS, apply_style, add_panel_label,
    DOUBLE_COL, SINGLE_COL, save_figure,
)


class ModelBenchmarkPlot:
    """Benchmark visualisation for scoring models.

    Usage:
        plot = ModelBenchmarkPlot()
        fig = plot.plot_spearman_comparison(results)
        fig = plot.plot_predicted_vs_measured(predicted, measured, model_name)
        fig = plot.plot_benchmark_panel(all_results)
    """

    def plot_spearman_comparison(
        self,
        results: dict[str, float],
        title: str = "Model comparison — Spearman ρ",
    ) -> plt.Figure:
        """Bar chart comparing Spearman correlation across models.

        Standard figure in guide prediction papers (NAR, Nature Biotech).

        Args:
            results: {model_name: spearman_rho} e.g.
                     {"Heuristic": 0.42, "Seq-CNN": 0.58, "JEPA-A": 0.67}
        """
        apply_style()

        models = list(results.keys())
        rhos = list(results.values())
        n = len(models)

        fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.8))

        colors = MODEL_COLORS[:n]
        bars = ax.bar(range(n), rhos, color=colors, edgecolor="white",
                      width=0.65, linewidth=0.5)

        # Value labels
        for bar, rho in zip(bars, rhos):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{rho:.3f}", ha="center", va="bottom", fontsize=5.5,
                fontweight="bold", color=PALETTE["dark"],
            )

        ax.set_xticks(range(n))
        ax.set_xticklabels(models, fontsize=6, rotation=20, ha="right")
        ax.set_ylabel("Spearman ρ", fontsize=7)
        ax.set_ylim(0, min(1.0, max(rhos) + 0.15))
        ax.set_title(title, fontsize=8, fontweight="bold")

        # Reference line at 0.5
        ax.axhline(0.5, color=PALETTE["grey"], linestyle=":", linewidth=0.6, alpha=0.6)

        fig.tight_layout()
        return fig

    def plot_predicted_vs_measured(
        self,
        predicted: np.ndarray,
        measured: np.ndarray,
        model_name: str,
        color: Optional[str] = None,
    ) -> plt.Figure:
        """Scatter plot of predicted vs measured efficiency with regression line.

        The canonical figure for guide prediction benchmarking.

        Args:
            predicted: Model predictions (0-1 scale).
            measured: Experimental measurements (0-1 scale).
            model_name: e.g. "JEPA Path A"
        """
        apply_style()

        if color is None:
            color = PALETTE["blue"]

        rho, p_val = stats.spearmanr(predicted, measured)

        fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))

        ax.scatter(measured, predicted, c=color, s=8, alpha=0.5,
                   edgecolors="none", rasterized=True)

        # Regression line
        z = np.polyfit(measured, predicted, 1)
        x_line = np.linspace(0, 1, 100)
        ax.plot(x_line, np.polyval(z, x_line), color=PALETTE["red"],
                linewidth=1, linestyle="-", alpha=0.8)

        # Diagonal reference
        ax.plot([0, 1], [0, 1], color=PALETTE["grey"], linewidth=0.6,
                linestyle="--", alpha=0.5)

        # Stats annotation
        ax.text(
            0.05, 0.92,
            f"ρ = {rho:.3f}\nn = {len(predicted)}",
            transform=ax.transAxes, fontsize=6,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=PALETTE["light_blue"],
                      edgecolor=PALETTE["border"], alpha=0.8),
        )

        ax.set_xlabel("Measured efficiency", fontsize=7)
        ax.set_ylabel("Predicted efficiency", fontsize=7)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.set_title(f"Prediction accuracy — {model_name}",
                     fontsize=8, fontweight="bold")

        fig.tight_layout()
        return fig

    def plot_benchmark_panel(
        self,
        all_results: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> plt.Figure:
        """Multi-panel scatter: one subplot per model, side by side.

        Args:
            all_results: {model_name: (predicted, measured)} for each model.
        """
        apply_style()

        models = list(all_results.keys())
        n = len(models)

        fig, axes = plt.subplots(
            1, n, figsize=(DOUBLE_COL, DOUBLE_COL / n * 0.85),
            sharey=True, gridspec_kw={"wspace": 0.12},
        )
        if n == 1:
            axes = [axes]

        panel_labels = "abcdefgh"

        for i, (name, (pred, meas)) in enumerate(all_results.items()):
            ax = axes[i]
            color = MODEL_COLORS[i % len(MODEL_COLORS)]
            rho, _ = stats.spearmanr(pred, meas)

            ax.scatter(meas, pred, c=color, s=5, alpha=0.4,
                       edgecolors="none", rasterized=True)

            # Regression
            z = np.polyfit(meas, pred, 1)
            x_line = np.linspace(0, 1, 100)
            ax.plot(x_line, np.polyval(z, x_line), color=color,
                    linewidth=1, alpha=0.8)
            ax.plot([0, 1], [0, 1], color=PALETTE["grey"], linewidth=0.5,
                    linestyle="--", alpha=0.4)

            ax.text(0.05, 0.92, f"ρ = {rho:.3f}", transform=ax.transAxes,
                    fontsize=5.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=PALETTE["border"], alpha=0.9))

            ax.set_title(name, fontsize=7, fontweight="bold")
            ax.set_xlabel("Measured", fontsize=6)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect("equal")
            add_panel_label(ax, panel_labels[i])

        axes[0].set_ylabel("Predicted", fontsize=6)

        fig.tight_layout()
        return fig
