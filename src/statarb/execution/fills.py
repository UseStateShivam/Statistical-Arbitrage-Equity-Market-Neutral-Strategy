"""
Order fill simulation and execution price determination.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from statarb.config import ExecutionConfig
from statarb.execution.costs import TransactionCostModel


@dataclass
class FillRecord:
    fill_id: str
    date: pd.Timestamp
    symbol: str
    side: str  # "BUY" or "SELL"
    shares: float
    nominal_price: float
    effective_price: float
    traded_notional: float
    slippage_cost: float
    commission_cost: float
    total_cost: float


class ExecutionSimulator:
    """
    Simulates execution fills at day t+1 Open price, incorporating slippage and volume limits.
    """

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.cost_model = TransactionCostModel(config)

    def execute_rebalance(
        self,
        date: pd.Timestamp,
        current_shares: dict[str, float],
        target_dollar_allocations: dict[str, float],
        open_prices: dict[str, float],
        trailing_adv_shares: dict[str, float] | None = None,
    ) -> list[FillRecord]:
        """
        Executes orders required to transition current positions to target allocations.
        Trades occur at day t+1 Open prices.
        """
        fills: list[FillRecord] = []
        all_symbols = set(current_shares.keys()).union(set(target_dollar_allocations.keys()))

        slippage_rate = self.config.slippage_bps / 10000.0

        for sym in all_symbols:
            open_p = open_prices.get(sym, 0.0)
            if open_p <= 0:
                continue

            curr_sh = current_shares.get(sym, 0.0)
            target_dollars = target_dollar_allocations.get(sym, 0.0)
            target_sh = target_dollars / open_p

            shares_diff = target_sh - curr_sh
            if abs(shares_diff) < 1e-4:
                continue

            # Volume participation limit: clamp order if exceeding max % of ADV
            if trailing_adv_shares and sym in trailing_adv_shares:
                adv = trailing_adv_shares[sym]
                max_order_shares = adv * self.config.max_volume_pct
                if abs(shares_diff) > max_order_shares:
                    shares_diff = np.sign(shares_diff) * max_order_shares

            side = "BUY" if shares_diff > 0 else "SELL"
            executed_shares = abs(shares_diff)
            traded_notional = executed_shares * open_p

            # Effective price after modeled slippage
            if side == "BUY":
                effective_p = open_p * (1.0 + slippage_rate)
            else:
                effective_p = open_p * (1.0 - slippage_rate)

            cost_breakdown = self.cost_model.compute_execution_cost(
                traded_notional=traded_notional, shares_traded=executed_shares
            )

            fill = FillRecord(
                fill_id=f"FILL_{date.strftime('%Y%m%d')}_{sym}_{side}",
                date=date,
                symbol=sym,
                side=side,
                shares=executed_shares if side == "BUY" else -executed_shares,
                nominal_price=open_p,
                effective_price=effective_p,
                traded_notional=traded_notional,
                slippage_cost=cost_breakdown.slippage_cost,
                commission_cost=cost_breakdown.commission_cost,
                total_cost=cost_breakdown.total_execution_cost,
            )
            fills.append(fill)

        return fills
