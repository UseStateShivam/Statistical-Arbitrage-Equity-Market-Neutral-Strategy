"""
Unit tests for quantitative performance, risk, and turnover metrics.
"""

import numpy as np
import pandas as pd

from statarb.metrics.performance import (
    compute_cagr,
    compute_cumulative_return,
)
from statarb.metrics.risk import compute_max_drawdown, compute_sharpe_ratio, compute_sortino_ratio
from statarb.metrics.turnover import compute_annualized_turnover


def test_cumulative_return_and_cagr():
    dates = pd.bdate_range("2021-01-01", periods=252)
    equity = pd.Series(np.linspace(100.0, 110.0, 252), index=dates)

    cum_ret = compute_cumulative_return(equity)
    assert np.isclose(cum_ret, 0.10)

    cagr = compute_cagr(equity, annualization_factor=252)
    assert np.isclose(cagr, 0.10, atol=0.01)


def test_sharpe_and_sortino():
    # 252 days with mean daily return 0.001 and daily std 0.01
    np.random.seed(42)
    r = pd.Series(np.random.normal(0.001, 0.01, 252))
    sharpe = compute_sharpe_ratio(r)

    # Theoretical Sharpe = sqrt(252) * 0.1 ~ 1.58
    assert 1.0 < sharpe < 2.2

    sortino = compute_sortino_ratio(r)
    assert sortino > 0.0


def test_max_drawdown():
    equity = pd.Series([100.0, 120.0, 96.0, 110.0, 130.0])
    max_dd, _max_duration = compute_max_drawdown(equity)

    # Peak was 120, trough was 96 -> DD = (96 - 120)/120 = -20% -> max_dd = 0.20
    assert np.isclose(max_dd, 0.20)


def test_annualized_turnover():
    dates = pd.bdate_range("2022-01-01", periods=100)
    notional = pd.Series(10000.0, index=dates)
    equity = pd.Series(100000.0, index=dates)

    # daily traded notional / equity = 10,000 / 100,000 = 0.10
    # Annualized two-way = 252 * 0.10 = 25.2x
    stats = compute_annualized_turnover(notional, equity)
    assert np.isclose(stats["annualized_two_way_turnover"], 25.2)
    assert np.isclose(stats["annualized_one_way_turnover"], 12.6)
