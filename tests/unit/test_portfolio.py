"""
Unit tests for portfolio sizing, exposure tracking, and constraints.
"""

import numpy as np

from statarb.config import PortfolioConfig
from statarb.portfolio.constraints import PortfolioConstraintManager
from statarb.portfolio.exposure import compute_portfolio_exposure
from statarb.portfolio.sizing import compute_pair_risk_weights


def test_inverse_volatility_sizing():
    cfg = PortfolioConfig(max_capital_per_pair=1.0, sizing_method="inv_vol")
    # Pair A has half the volatility of Pair B -> Pair A gets 2x weight of Pair B
    stds = {"PAIR_LOW_VOL": 0.01, "PAIR_HIGH_VOL": 0.02}
    weights = compute_pair_risk_weights(stds, cfg)

    assert weights["PAIR_LOW_VOL"] > weights["PAIR_HIGH_VOL"]
    assert np.isclose(weights["PAIR_LOW_VOL"] / weights["PAIR_HIGH_VOL"], 2.0, atol=0.01)


def test_portfolio_exposure_metrics():
    holdings = {"AAPL": 50000.0, "MSFT": -50000.0}
    equity = 100000.0
    betas = {"AAPL": 1.1, "MSFT": 1.0}

    exp = compute_portfolio_exposure(holdings, equity, betas)
    assert np.isclose(exp.gross_exposure_pct, 1.0)
    assert np.isclose(exp.net_exposure_pct, 0.0)  # Dollar neutral
    assert np.isclose(exp.long_exposure_pct, 0.5)
    assert np.isclose(exp.short_exposure_pct, 0.5)
    assert np.isclose(exp.portfolio_beta, 0.05)  # 0.5 * 1.1 - 0.5 * 1.0


def test_constraints_single_stock_and_gross():
    cfg = PortfolioConfig(max_single_stock_weight=0.15, max_gross_exposure=1.0)
    cm = PortfolioConstraintManager(cfg)

    # Propose allocation with 30% in AAPL (violates 15% single stock cap)
    allocs = {"AAPL": 300000.0, "MSFT": -100000.0}
    equity = 1000000.0

    constrained = cm.apply_constraints(allocs, equity)
    assert abs(constrained["AAPL"]) <= 150000.0  # Capped at 15% ($150,000)
