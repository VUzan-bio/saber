"""SABER visualisation module.

Publication-quality figures for CRISPR-Cas12a crRNA design analysis.
All plots follow Nature Methods / NAR figure conventions:
- Matplotlib with custom rcParams (no seaborn defaults)
- Two-column journal width (180 mm) or single-column (88 mm)
- Consistent colour palette across all figures
- Vector output (PDF/SVG) by default
"""

from saber.viz.style import apply_style, PALETTE, save_figure
from saber.viz.discrimination import DiscriminationHeatmap
from saber.viz.ranking import CandidateRankingPlot
from saber.viz.multiplex import MultiplexMatrixPlot
from saber.viz.benchmark import ModelBenchmarkPlot
from saber.viz.active_learning import ActiveLearningPlot
from saber.viz.target_overview import TargetDashboard

__all__ = [
    "apply_style",
    "PALETTE",
    "save_figure",
    "DiscriminationHeatmap",
    "CandidateRankingPlot",
    "MultiplexMatrixPlot",
    "ModelBenchmarkPlot",
    "ActiveLearningPlot",
    "TargetDashboard",
]
