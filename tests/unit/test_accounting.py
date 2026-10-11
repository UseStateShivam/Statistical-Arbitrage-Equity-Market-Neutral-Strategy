"""
Unit tests for portfolio accounting and double-entry ledger reconciliation.
"""

import numpy as np
import pandas as pd

from statarb.backtest.accounting import reconcile_ledger
from statarb.execution.fills import FillRecord
from statarb.execution.ledger import PortfolioLedger


def test_ledger_fill_application_and_mark_to_market():
    ledger = PortfolioLedger(initial_capital=100000.0)

    # Buy 100 shares of XYZ at $100 + $5 slippage + $0.10 commission
    fill = FillRecord(
        fill_id="FILL_1",
        date=pd.Timestamp("2022-01-05"),
        symbol="XYZ",
        side="BUY",
        shares=100.0,
        nominal_price=100.0,
        effective_price=100.05,
        traded_notional=10000.0,
        slippage_cost=5.0,
        commission_cost=0.10,
        total_cost=5.10,
    )
    ledger.apply_fills([fill])

    assert ledger.shares["XYZ"] == 100.0
    # Cash spent = 100 * 100.05 + 0.10 = 10005.10
    assert np.isclose(ledger.cash, 100000.0 - 10005.10)

    # Mark to market at close price $105
    snap = ledger.mark_to_market(
        date=pd.Timestamp("2022-01-05"),
        close_prices={"XYZ": 105.0},
        traded_notional=10000.0,
        transaction_costs=5.10,
    )

    # Holdings MV = 100 * 105 = 10500.0
    assert snap.holdings_market_val == 10500.0
    # Equity = Cash + Holdings MV
    assert np.isclose(snap.portfolio_equity, ledger.cash + 10500.0)

    # Full ledger reconciliation
    is_rec, stats = reconcile_ledger(ledger)
    assert is_rec
    assert stats["discrepancy"] < 0.01
