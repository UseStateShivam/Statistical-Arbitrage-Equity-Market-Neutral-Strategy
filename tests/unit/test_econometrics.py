"""
Unit tests for econometric procedures: ADF, Engle-Granger, Johansen, and FDR.
"""

import numpy as np

from statarb.research.cointegration import (
    apply_benjamini_hochberg,
    run_engle_granger,
    run_johansen,
)
from statarb.research.hedge_ratio import estimate_ols_hedge_ratio, estimate_tls_hedge_ratio
from statarb.research.stationarity import run_adf_test


def test_adf_test_stationary_vs_random_walk(stationary_spread, random_walk_prices):
    # 1. Stationary OU process must reject H0
    res_stat = run_adf_test(stationary_spread)
    assert res_stat.is_stationary
    assert res_stat.pvalue < 0.05

    # 2. Random walk must NOT reject H0
    rw1, _ = random_walk_prices
    res_rw = run_adf_test(rw1)
    assert not res_rw.is_stationary
    assert res_rw.pvalue > 0.10


def test_engle_granger_cointegration(cointegrated_prices, random_walk_prices):
    pa_coint, pb_coint = cointegrated_prices
    log_a = np.log(pa_coint)
    log_b = np.log(pb_coint)

    eg_res = run_engle_granger(log_a, log_b)
    assert eg_res.is_cointegrated
    assert eg_res.pvalue < 0.05
    assert 0.9 < eg_res.beta < 1.3  # True beta = 1.1

    # Independent random walks must fail cointegration
    rw1, rw2 = random_walk_prices
    eg_rw = run_engle_granger(np.log(rw1), np.log(rw2))
    assert not eg_rw.is_cointegrated


def test_johansen_cointegration_rank(cointegrated_prices, random_walk_prices):
    pa_coint, pb_coint = cointegrated_prices
    joh_res = run_johansen(np.log(pa_coint), np.log(pb_coint))
    assert joh_res.cointegration_rank == 1
    assert joh_res.is_cointegrated

    rw1, rw2 = random_walk_prices
    joh_rw = run_johansen(np.log(rw1), np.log(rw2))
    assert joh_rw.cointegration_rank == 0
    assert not joh_rw.is_cointegrated


def test_benjamini_hochberg_fdr():
    # Mix of significant and insignificant p-values
    pvals = [0.001, 0.004, 0.012, 0.045, 0.35, 0.82]
    rejected, qvals = apply_benjamini_hochberg(pvals, alpha=0.05)

    assert len(rejected) == len(pvals)
    # Monotonicity check
    assert qvals[0] < qvals[-1]
    assert rejected[0]  # Very small p-value rejected
    assert not rejected[-1]  # 0.82 not rejected


def test_hedge_ratio_ols_and_tls():
    x = np.linspace(1, 10, 200)
    y = 0.5 + 2.0 * x + np.random.normal(0, 0.1, 200)

    ols = estimate_ols_hedge_ratio(y, x)
    assert np.isclose(ols.beta, 2.0, atol=0.05)
    assert np.isclose(ols.alpha, 0.5, atol=0.2)
    assert ols.r_squared > 0.95

    tls = estimate_tls_hedge_ratio(y, x)
    assert np.isclose(tls.beta, 2.0, atol=0.05)
