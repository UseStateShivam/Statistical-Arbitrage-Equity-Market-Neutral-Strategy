"""
Candidate pair generation, sector clustering, and computational pre-filtering.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from statarb.config import CandidateGenerationConfig


@dataclass
class CandidatePair:
    pair_id: str
    asset_a: str
    asset_b: str
    sector: str
    correlation: float


class CandidatePairGenerator:
    """
    Generates candidate equity pairs within sectors based on economic links
    and correlation pre-screening.
    """

    def __init__(self, config: CandidateGenerationConfig, sector_map: dict[str, str] | None = None):
        self.config = config
        self.sector_map = sector_map or {}

    def generate_candidates(self, returns_panel: pd.DataFrame) -> list[CandidatePair]:
        """
        Generates candidate pairs from returns panel over the formation window.
        """
        symbols = [s for s in returns_panel.columns if s.upper() != "SPY"]
        candidates: list[CandidatePair] = []

        # Calculate pairwise Pearson correlation matrix
        corr_matrix = returns_panel[symbols].corr()

        # Group symbols by sector if sector map provided
        if self.config.same_sector_only and self.sector_map:
            sector_groups: dict[str, list[str]] = {}
            for sym in symbols:
                sec = self.sector_map.get(sym, "Unknown")
                sector_groups.setdefault(sec, []).append(sym)
        else:
            sector_groups = {"All": symbols}

        for sector, sec_symbols in sector_groups.items():
            if len(sec_symbols) < 2:
                continue

            sector_candidates = []
            for a, b in combinations(sec_symbols, 2):
                corr_val = corr_matrix.loc[a, b]
                if np.isnan(corr_val):
                    continue

                if corr_val >= self.config.min_correlation:
                    # Deterministic ordering: alphabetical
                    asset_a, asset_b = (a, b) if a < b else (b, a)
                    pair_id = f"{sector}_{asset_a}_{asset_b}"
                    sector_candidates.append(
                        CandidatePair(
                            pair_id=pair_id,
                            asset_a=asset_a,
                            asset_b=asset_b,
                            sector=sector,
                            correlation=float(corr_val),
                        )
                    )

            # Sort by correlation descending and cap per sector
            sector_candidates.sort(key=lambda x: x.correlation, reverse=True)
            candidates.extend(sector_candidates[: self.config.max_candidates_per_sector])

        return candidates
