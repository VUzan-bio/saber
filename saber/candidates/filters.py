"""Hard filters for crRNA candidates.

Applied after PAM scanning to remove candidates that violate biophysical
constraints. These are binary pass/fail — no partial credit.

Filter order matters for performance: cheapest filters first.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from saber.core.constants import (
    GC_MAX,
    GC_MIN,
    HOMOPOLYMER_MAX,
    MFE_THRESHOLD,
    SEED_REGION_END,
)
from saber.core.types import CrRNACandidate

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Why a candidate was kept or rejected."""
    passed: bool
    reason: Optional[str] = None


class CandidateFilter:
    """Apply hard filters to crRNA candidates.

    Usage:
        filt = CandidateFilter(require_seed=True, check_structure=True)
        passed = filt.filter_batch(candidates)
    """

    def __init__(
        self,
        gc_min: float = GC_MIN,
        gc_max: float = GC_MAX,
        homopolymer_max: int = HOMOPOLYMER_MAX,
        mfe_threshold: float = MFE_THRESHOLD,
        require_seed: bool = True,
        check_structure: bool = True,
    ) -> None:
        self.gc_min = gc_min
        self.gc_max = gc_max
        self.homopolymer_max = homopolymer_max
        self.mfe_threshold = mfe_threshold
        self.require_seed = require_seed
        self.check_structure = check_structure

    def apply(self, candidate: CrRNACandidate) -> FilterResult:
        """Evaluate a single candidate against all filters."""
        # 1. Seed region check (cheapest)
        if self.require_seed and not candidate.in_seed:
            return FilterResult(False, f"mutation at position {candidate.mutation_position_in_spacer}, outside seed (1-{SEED_REGION_END})")

        # 2. GC content
        if not (self.gc_min <= candidate.gc_content <= self.gc_max):
            return FilterResult(False, f"GC={candidate.gc_content:.2f}, required [{self.gc_min}-{self.gc_max}]")

        # 3. Homopolymer
        if candidate.homopolymer_max > self.homopolymer_max:
            return FilterResult(False, f"homopolymer run={candidate.homopolymer_max}, max={self.homopolymer_max}")

        # 4. Secondary structure (expensive — last)
        if self.check_structure:
            mfe = self._compute_mfe(candidate.spacer_seq)
            if mfe is not None:
                candidate.mfe = mfe
                if mfe < self.mfe_threshold:
                    return FilterResult(False, f"MFE={mfe:.1f} kcal/mol, threshold={self.mfe_threshold}")

        return FilterResult(True)

    def filter_batch(self, candidates: list[CrRNACandidate]) -> list[CrRNACandidate]:
        """Filter a list, returning only those that pass."""
        passed: list[CrRNACandidate] = []
        rejected = 0

        for c in candidates:
            result = self.apply(c)
            if result.passed:
                passed.append(c)
            else:
                rejected += 1
                logger.debug("Rejected %s: %s", c.candidate_id, result.reason)

        logger.info(
            "Filtering: %d passed, %d rejected (%.0f%% pass rate)",
            len(passed), rejected,
            100 * len(passed) / len(candidates) if candidates else 0,
        )
        return passed

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_mfe(spacer: str) -> Optional[float]:
        """Compute minimum free energy using ViennaRNA RNAfold.

        Returns None if ViennaRNA is not installed.
        """
        try:
            result = subprocess.run(
                ["RNAfold", "--noPS"],
                input=spacer,
                capture_output=True,
                text=True,
                timeout=10,
            )
            # Output format: "sequence\nstructure (MFE)"
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                # Extract MFE from parenthetical at end of line 2
                mfe_str = lines[1].split("(")[-1].rstrip(")")
                return float(mfe_str.strip())
        except FileNotFoundError:
            logger.debug("ViennaRNA not installed, skipping MFE filter")
        except (subprocess.TimeoutExpired, ValueError, IndexError):
            logger.warning("RNAfold failed for spacer: %s", spacer)

        return None
