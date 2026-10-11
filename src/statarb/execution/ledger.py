"""
Auditable portfolio ledger, cash accounting, and mark-to-market tracking.
"""

from dataclasses import dataclass

import pandas as pd

from statarb.execution.fills import FillRecord


@dataclass
class DailySnapshot:
    date: pd.Timestamp
    cash: float
    holdings_market_val: float
    portfolio_equity: float
    gross_market_val: float
    net_market_val: float
    daily_traded_notional: float
    daily_transaction_costs: float
    daily_borrow_costs: float
    daily_cash_interest: float
    daily_gross_pnl: float
    daily_net_pnl: float
    daily_net_return: float


class PortfolioLedger:
    """
    Maintains complete double-entry cash and asset records across backtest sessions.
    """

    def __init__(self, initial_capital: float = 1000000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.shares: dict[str, float] = {}  # symbol -> share count (signed)
        self.fills_history: list[FillRecord] = []
        self.snapshots: list[DailySnapshot] = []
        self.cumulative_costs = 0.0

    def apply_fills(self, fills: list[FillRecord]) -> None:
        """
        Updates cash and share positions from executed trade fills.
        """
        for fill in fills:
            self.fills_history.append(fill)

            # Cash flow from trade
            # BUY: pay notional + commission
            # SELL: receive notional - commission
            if fill.side == "BUY":
                cash_delta = -(abs(fill.shares) * fill.effective_price + fill.commission_cost)
                self.shares[fill.symbol] = self.shares.get(fill.symbol, 0.0) + abs(fill.shares)
            else:  # SELL
                cash_delta = +(abs(fill.shares) * fill.effective_price - fill.commission_cost)
                self.shares[fill.symbol] = self.shares.get(fill.symbol, 0.0) - abs(fill.shares)

            self.cash += cash_delta
            self.cumulative_costs += fill.total_cost

            # Clean tiny residual shares
            if abs(self.shares.get(fill.symbol, 0.0)) < 1e-4:
                self.shares.pop(fill.symbol, None)

    def mark_to_market(
        self,
        date: pd.Timestamp,
        close_prices: dict[str, float],
        daily_borrow_cost: float = 0.0,
        daily_cash_interest: float = 0.0,
        traded_notional: float = 0.0,
        transaction_costs: float = 0.0,
    ) -> DailySnapshot:
        """
        Performs session-end mark-to-market valuation and records daily snapshot.
        """
        # Apply overnight borrow fee and cash yield
        self.cash -= daily_borrow_cost
        self.cash += daily_cash_interest

        long_mv = 0.0
        short_mv = 0.0

        for sym, sh in self.shares.items():
            price = close_prices.get(sym, 0.0)
            mv = sh * price
            if mv > 0:
                long_mv += mv
            else:
                short_mv += mv

        holdings_mv = long_mv + short_mv
        portfolio_equity = self.cash + holdings_mv

        gross_mv = long_mv + abs(short_mv)
        net_mv = holdings_mv

        # Prior equity
        prior_equity = (
            self.snapshots[-1].portfolio_equity if self.snapshots else self.initial_capital
        )
        daily_net_pnl = portfolio_equity - prior_equity
        daily_net_return = (daily_net_pnl / prior_equity) if prior_equity > 0 else 0.0
        daily_gross_pnl = (
            daily_net_pnl + transaction_costs + daily_borrow_cost - daily_cash_interest
        )

        snapshot = DailySnapshot(
            date=date,
            cash=self.cash,
            holdings_market_val=holdings_mv,
            portfolio_equity=portfolio_equity,
            gross_market_val=gross_mv,
            net_market_val=net_mv,
            daily_traded_notional=traded_notional,
            daily_transaction_costs=transaction_costs,
            daily_borrow_costs=daily_borrow_cost,
            daily_cash_interest=daily_cash_interest,
            daily_gross_pnl=daily_gross_pnl,
            daily_net_pnl=daily_net_pnl,
            daily_net_return=daily_net_return,
        )
        self.snapshots.append(snapshot)
        return snapshot

    def to_dataframe(self) -> pd.DataFrame:
        """Converts snapshots to time-indexed DataFrame."""
        if not self.snapshots:
            return pd.DataFrame()
        records = [s.__dict__ for s in self.snapshots]
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")
