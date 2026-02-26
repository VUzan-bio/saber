"""Shared style for all SABER figures.

Conventions follow Nature Methods / Nucleic Acids Research:
- Arial/Helvetica font family
- Minimal chartjunk (no top/right spines, no grid unless essential)
- Accessible colour palette (colourblind-safe)
- Consistent sizing for single-column (88 mm) and two-column (180 mm) figures
- 300 DPI minimum for raster, vector preferred
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Colour palette — colourblind-safe, muted, publication-ready
# ---------------------------------------------------------------------------

PALETTE = {
    # Primary
    "blue": "#2C6FAC",
    "red": "#C0392B",
    "green": "#27AE60",
    "orange": "#E67E22",
    "purple": "#8E44AD",
    "teal": "#16A085",

    # Muted / fills
    "light_blue": "#DAEAF6",
    "light_red": "#FADBD8",
    "light_green": "#D5F5E3",
    "light_orange": "#FDEBD0",
    "light_grey": "#F2F3F4",

    # Heatmap endpoints
    "heatmap_low": "#FFFFFF",
    "heatmap_high": "#2C6FAC",
    "diverging_low": "#2166AC",
    "diverging_mid": "#F7F7F7",
    "diverging_high": "#B2182B",

    # Text
    "dark": "#2C3E50",
    "grey": "#7F8C8D",
    "border": "#BDC3C7",
}

# Sequential colourmap for heatmaps
CMAP_SEQ = mpl.colors.LinearSegmentedColormap.from_list(
    "saber_seq", [PALETTE["heatmap_low"], PALETTE["light_blue"], PALETTE["blue"]]
)

# Diverging colourmap for discrimination (blue = low activity, red = high)
CMAP_DIV = mpl.colors.LinearSegmentedColormap.from_list(
    "saber_div", [PALETTE["diverging_low"], PALETTE["diverging_mid"], PALETTE["diverging_high"]]
)

# Categorical colours for model comparison
MODEL_COLORS = [
    PALETTE["blue"], PALETTE["red"], PALETTE["green"],
    PALETTE["orange"], PALETTE["purple"], PALETTE["teal"],
]

# ---------------------------------------------------------------------------
# Figure sizing (in mm → inches)
# ---------------------------------------------------------------------------

MM_TO_INCH = 1 / 25.4

SINGLE_COL = 88 * MM_TO_INCH       # ~3.46 inches
DOUBLE_COL = 180 * MM_TO_INCH      # ~7.09 inches
QUARTER_PAGE = (88 * MM_TO_INCH, 66 * MM_TO_INCH)
HALF_PAGE = (180 * MM_TO_INCH, 80 * MM_TO_INCH)
FULL_PAGE = (180 * MM_TO_INCH, 220 * MM_TO_INCH)


def apply_style() -> None:
    """Apply SABER publication style to all matplotlib figures."""
    mpl.rcParams.update({
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,

        # Axes
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": PALETTE["dark"],
        "axes.labelcolor": PALETTE["dark"],
        "axes.titleweight": "bold",

        # Ticks
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",

        # Lines
        "lines.linewidth": 1.0,
        "lines.markersize": 4,

        # Legend
        "legend.frameon": False,
        "legend.borderpad": 0.3,
        "legend.handlelength": 1.5,

        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,

        # Grid (off by default)
        "axes.grid": False,
    })


def save_figure(
    fig: plt.Figure,
    name: str,
    output_dir: str | Path = "figures",
    formats: tuple[str, ...] = ("pdf", "png"),
) -> list[Path]:
    """Save figure in multiple formats."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = out / f"{name}.{fmt}"
        fig.savefig(p, format=fmt, dpi=300 if fmt == "png" else None)
        paths.append(p)
    plt.close(fig)
    return paths


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.08) -> None:
    """Add panel label (a, b, c...) in Nature style — bold, upper-left."""
    ax.text(
        x, y, label, transform=ax.transAxes,
        fontsize=10, fontweight="bold", va="top", ha="left",
        color=PALETTE["dark"],
    )
