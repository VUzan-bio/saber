"""Candidate ranking visualisation.

Horizontal stacked bar charts showing score decomposition for top-N candidates
per target. Directly supports experimental decision-making: which candidates
to test first and why.

Inspired by:
- CRISPR guide scoring tool outputs (CRISPRon, CHOPCHOP)
- Feature importance plots from Kim et al. 2018
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


class CandidateRankingPlot:
    """Visualise ranked crRNA candidates with score breakdown.

    Usage:
        plot = CandidateRankingPlot()
        fig = plot.plot_ranking(candidates_data, target_label)
        fig = plot.plot_score_radar(score_dict, candidate_id)
        fig = plot.plot_multi_target_top(top_per_target)
    """

    COMPONENT_COLORS = {
        "seed_position": PALETTE["blue"],
        "gc": PALETTE["teal"],
        "structure": PALETTE["purple"],
        "homopolymer": PALETTE["orange"],
        "offtarget": PALETTE["red"],
    }

    COMPONENT_LABELS = {
        "seed_position": "Seed position",
        "gc": "GC content",
        "structure": "2° structure",
        "homopolymer": "Homopolymer",
        "offtarget": "Off-target",
    }

    def plot_ranking(
        self,
        candidates: list[dict],
        target_label: str,
        top_n: int = 15,
        show_components: bool = True,
    ) -> plt.Figure:
        """Horizontal bar chart of top-N candidates with score breakdown.

        Args:
            candidates: List of dicts with keys:
                - candidate_id: str
                - spacer_seq: str
                - composite: float (total score)
                - seed_position_score, gc_penalty, structure_penalty,
                  homopolymer_penalty, offtarget_penalty: float (sub-scores)
                - ml_score: Optional[float]
            target_label: e.g. "rpoB_S531L"
            top_n: Number of candidates to show.
        """
        apply_style()

        data = sorted(candidates, key=lambda x: x["composite"], reverse=True)[:top_n]
        n = len(data)

        fig, ax = plt.subplots(figsize=(DOUBLE_COL, max(2.5, n * 0.22)))

        if show_components:
            components = ["seed_position", "gc", "structure", "homopolymer", "offtarget"]
            weights = {"seed_position": 0.35, "gc": 0.20, "structure": 0.20,
                       "homopolymer": 0.10, "offtarget": 0.15}

            lefts = np.zeros(n)
            for comp in components:
                key = f"{comp}_score" if comp == "seed_position" else f"{comp}_penalty"
                vals = np.array([d.get(key, d.get(comp, 0)) * weights[comp] for d in data])
                ax.barh(
                    range(n), vals, left=lefts, height=0.7,
                    color=self.COMPONENT_COLORS[comp],
                    edgecolor="white", linewidth=0.3,
                    label=self.COMPONENT_LABELS[comp],
                )
                lefts += vals

            ax.legend(
                loc="lower right", ncol=2, fontsize=5,
                handlelength=1, handletextpad=0.4,
                columnspacing=0.8,
            )
        else:
            composites = [d["composite"] for d in data]
            colors = [PALETTE["blue"] if i == 0 else PALETTE["light_blue"]
                      for i in range(n)]
            ax.barh(range(n), composites, color=colors, edgecolor="white", height=0.7)

        # Labels
        labels = []
        for d in data:
            seq = d.get("spacer_seq", d.get("candidate_id", ""))
            short_seq = seq[:10] + "..." if len(seq) > 13 else seq
            labels.append(f"#{data.index(d)+1}  {short_seq}")

        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=5, fontfamily="monospace")
        ax.set_xlabel("Composite score", fontsize=7)
        ax.set_title(f"crRNA candidates — {target_label}", fontsize=8, fontweight="bold")
        ax.set_xlim(0, 1.0)
        ax.invert_yaxis()

        # ML score overlay if available
        for i, d in enumerate(data):
            ml = d.get("ml_score")
            if ml is not None:
                ax.plot(ml, i, marker="D", color=PALETTE["red"],
                        markersize=3, zorder=5)

        fig.tight_layout()
        return fig

    def plot_score_radar(
        self,
        scores: dict[str, float],
        candidate_id: str,
    ) -> plt.Figure:
        """Radar/spider plot of sub-score components for a single candidate.

        Classic diagnostic figure showing the "shape" of a candidate's
        strengths and weaknesses.
        """
        apply_style()

        categories = list(scores.keys())
        values = list(scores.values())
        n = len(categories)

        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL),
                               subplot_kw=dict(polar=True))

        ax.fill(angles, values, color=PALETTE["blue"], alpha=0.15)
        ax.plot(angles, values, color=PALETTE["blue"], linewidth=1.2)
        ax.scatter(angles[:-1], values[:-1], color=PALETTE["blue"], s=15, zorder=5)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=5)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=4, color=PALETTE["grey"])
        ax.spines["polar"].set_visible(False)

        ax.set_title(f"Score profile — {candidate_id}", fontsize=7,
                     fontweight="bold", pad=15)

        fig.tight_layout()
        return fig

    def plot_multi_target_top(
        self,
        top_per_target: dict[str, dict],
    ) -> plt.Figure:
        """Summary: best candidate per target across the full panel.

        Shows composite score + ML score (if available) for the rank-1
        candidate of each target. Colour-coded by validation status.
        """
        apply_style()

        targets = list(top_per_target.keys())
        n = len(targets)
        composites = [top_per_target[t].get("composite", 0) for t in targets]
        statuses = [top_per_target[t].get("status", "untested") for t in targets]

        status_colors = {
            "untested": PALETTE["light_blue"],
            "in_progress": PALETTE["light_orange"],
            "validated": PALETTE["green"],
            "failed": PALETTE["red"],
        }
        colors = [status_colors.get(s, PALETTE["light_grey"]) for s in statuses]

        fig, ax = plt.subplots(figsize=(DOUBLE_COL, max(2.0, n * 0.2)))

        ax.barh(range(n), composites, color=colors, edgecolor=PALETTE["border"],
                height=0.65, linewidth=0.5)

        ax.set_yticks(range(n))
        ax.set_yticklabels(targets, fontsize=6)
        ax.set_xlabel("Top candidate score", fontsize=7)
        ax.set_title("Panel overview — best candidate per target",
                     fontsize=8, fontweight="bold")
        ax.set_xlim(0, 1.0)
        ax.invert_yaxis()

        # Legend for status colours
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=c, edgecolor=PALETTE["border"], label=s.replace("_", " ").title())
            for s, c in status_colors.items()
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=5, ncol=2)

        fig.tight_layout()
        return fig
