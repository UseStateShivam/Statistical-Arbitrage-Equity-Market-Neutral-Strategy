"""
Transaction cost modeling: 5 bps slippage per traded side, commissions, and borrow fees.
"""

from dataclasses import dataclass

from statarb.config import ExecutionConfig


@dataclass
class TransactionCostBreakdown:
    traded_notional: float
    slippage_cost: float
    commission_cost: float
    total_execution_cost: float


class TransactionCostModel:
    """
    Computes trading frictions on actual executed trades and overnight short borrow fees.
    """

    def __init__(self, config: ExecutionConfig):
        self.config = config

    def compute_execution_cost(
        self, traded_notional: float, shares_traded: float
    ) -> TransactionCostBreakdown:
        """
        Calculates execution friction for a single transaction.
        Slippage: strictly 5 bps (0.0005) of traded notional per side.
        Commissions: per-share rate.
        """
        notional_abs = abs(traded_notional)
        slippage_rate = self.config.slippage_bps / 10000.0  # 5 bps -> 0.0005
        slippage = notional_abs * slippage_rate

        commissions = abs(shares_traded) * self.config.commission_per_share
        total = slippage + commissions

        return TransactionCostBreakdown(
            traded_notional=notional_abs,
            slippage_cost=slippage,
            commission_cost=commissions,
            total_execution_cost=total,
        )

    def compute_daily_borrow_cost(self, short_market_value: float) -> float:
        """
        Computes overnight financing borrow fee charged on short equity market value.
        annual_fee = 50 bps -> daily = 0.0050 / 252.
        """
        short_abs = abs(short_market_value)
        daily_rate = (self.config.borrow_fee_annual_bps / 10000.0) / 252.0
        return short_abs * daily_rate

    def compute_daily_cash_interest(self, cash_balance: float) -> float:
        """
        Credits overnight interest yield on unencumbered positive cash balance.
        """
        if cash_balance <= 0:
            return 0.0
        daily_rate = (self.config.cash_interest_annual_bps / 10000.0) / 252.0
        return cash_balance * daily_rate
