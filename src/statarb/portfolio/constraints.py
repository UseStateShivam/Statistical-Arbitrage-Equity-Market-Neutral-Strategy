"""
Portfolio risk constraints and concentration limits.
"""

import numpy as np

from statarb.config import PortfolioConfig


class PortfolioConstraintManager:
    """
    Enforces risk limits: gross exposure, single-stock concentration, and sector limits.
    """

    def __init__(self, config: PortfolioConfig, sector_map: dict[str, str] | None = None):
        self.config = config
        self.sector_map = sector_map or {}

    def apply_constraints(
        self, target_dollar_allocations: dict[str, float], portfolio_equity: float
    ) -> dict[str, float]:
        """
        Clamps and scales proposed target dollar allocations to strictly respect constraints.
        """
        if portfolio_equity <= 0 or not target_dollar_allocations:
            return {}

        allocs = target_dollar_allocations.copy()

        # 1. Enforce single stock concentration limit
        max_single_stock_dollars = self.config.max_single_stock_weight * portfolio_equity
        for sym, d_val in allocs.items():
            if abs(d_val) > max_single_stock_dollars:
                allocs[sym] = np.sign(d_val) * max_single_stock_dollars

        # 2. Enforce sector exposure limits
        if self.sector_map:
            sector_totals: dict[str, float] = {}
            for sym, d_val in allocs.items():
                sec = self.sector_map.get(sym, "Unknown")
                sector_totals[sec] = sector_totals.get(sec, 0.0) + abs(d_val)

            max_sector_dollars = self.config.max_sector_weight * portfolio_equity
            for sec, total_sec_val in sector_totals.items():
                if total_sec_val > max_sector_dollars:
                    scale = max_sector_dollars / total_sec_val
                    for sym in allocs:
                        if self.sector_map.get(sym, "Unknown") == sec:
                            allocs[sym] *= scale

        # 3. Enforce portfolio gross exposure limit
        total_gross = sum(abs(v) for v in allocs.values())
        max_gross_dollars = self.config.max_gross_exposure * portfolio_equity
        if total_gross > max_gross_dollars and total_gross > 0:
            gross_scale = max_gross_dollars / total_gross
            for sym in allocs:
                allocs[sym] *= gross_scale

        return allocs
