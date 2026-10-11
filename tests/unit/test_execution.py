"""
Unit tests for execution simulation, slippage model, and transaction costs.
"""

import numpy as np
import pandas as pd

from statarb.config import ExecutionConfig
from statarb.execution.costs import TransactionCostModel
from statarb.execution.fills import ExecutionSimulator


def test_baseline_5bps_slippage():
    cfg = ExecutionConfig(slippage_bps=5.0, commission_per_share=0.0)
    model = TransactionCostModel(cfg)

    # Required baseline test: $100,000 notional incurs exactly $50 slippage (0.05%)
    cost_res = model.compute_execution_cost(traded_notional=100000.0, shares_traded=1000.0)
    assert np.isclose(cost_res.slippage_cost, 50.0)
    assert np.isclose(cost_res.total_execution_cost, 50.0)


def test_short_borrow_cost():
    cfg = ExecutionConfig(borrow_fee_annual_bps=50.0)  # 50 bps annual
    model = TransactionCostModel(cfg)

    short_mv = -100000.0
    # Daily cost = 100,000 * (0.0050 / 252) ~ 1.9841
    daily_cost = model.compute_daily_borrow_cost(short_mv)
    assert np.isclose(daily_cost, 100000.0 * 0.0050 / 252.0)


def test_execution_simulator_generates_correct_fills():
    cfg = ExecutionConfig(slippage_bps=5.0, commission_per_share=0.001)
    sim = ExecutionSimulator(cfg)

    date = pd.Timestamp("2022-01-05")
    current_shares = {"AAPL": 0.0}
    target_dollars = {"AAPL": 15000.0}
    open_prices = {"AAPL": 150.0}

    fills = sim.execute_rebalance(
        date=date,
        current_shares=current_shares,
        target_dollar_allocations=target_dollars,
        open_prices=open_prices,
    )

    assert len(fills) == 1
    fill = fills[0]
    assert fill.symbol == "AAPL"
    assert fill.side == "BUY"
    assert np.isclose(fill.shares, 100.0)  # 15000 / 150
    assert np.isclose(fill.nominal_price, 150.0)
    assert fill.effective_price > 150.0  # Bought at higher price due to slippage
    assert np.isclose(fill.slippage_cost, 15000.0 * 0.0005)  # 5 bps
