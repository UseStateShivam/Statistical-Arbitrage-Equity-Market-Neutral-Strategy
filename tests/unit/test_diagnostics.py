"""
Unit tests for Hurst exponent, half-life, and structural stability diagnostics.
"""

import numpy as np

from statarb.research.diagnostics import run_chow_test
from statarb.research.half_life import estimate_half_life
from statarb.research.hurst import estimate_hurst_variance_time
from tests.fixtures.synthetic_processes import (
    generate_process_with_break,
    generate_stationary_ou_process,
)


def test_hurst_exponent_mean_reverting_vs_random_walk(stationary_spread, random_walk_prices):
    # OU process must be mean-reverting (H < 0.5)
    h_ou = estimate_hurst_variance_time(stationary_spread)
    assert h_ou.is_valid
    assert h_ou.hurst_exponent < 0.50
    assert h_ou.is_mean_reverting

    # Random walk
    rw1, _ = random_walk_prices
    h_rw = estimate_hurst_variance_time(rw1)
    assert h_ou.hurst_exponent < h_rw.hurst_exponent


def test_half_life_estimation():
    # Known OU theta = 0.05 -> theoretical half life = -ln(2) / ln(1 - 0.05) ~ 13.5 days
    spread = generate_stationary_ou_process(n=1000, theta=0.05, sigma=0.02, seed=42)
    hl_res = estimate_half_life(spread)

    assert hl_res.is_valid_mean_reverting
    assert 8.0 < hl_res.half_life_days < 20.0
    assert hl_res.coefficient < 0.0


def test_half_life_rejection_of_explosive_process():
    # Process with positive drift / explosive: S_t = S_{t-1} + 0.05 * S_{t-1} + noise (b >= 0)
    n = 200
    s = np.zeros(n)
    s[0] = 1.0
    for t in range(1, n):
        s[t] = s[t - 1] + 0.02 * s[t - 1] + np.random.normal(0, 0.01)

    hl_res = estimate_half_life(s)
    # MUST reject and NOT report a valid positive half-life
    assert not hl_res.is_valid_mean_reverting
    assert np.isnan(hl_res.half_life_days)
    assert hl_res.rejection_reason is not None
    assert "Non-mean-reverting coefficient" in hl_res.rejection_reason


def test_chow_test_detects_structural_break():
    pa, pb = generate_process_with_break(n=400, break_idx=200, seed=42)
    f_stat, p_val = run_chow_test(np.log(pa), np.log(pb), split_idx=200)

    # Significant F-stat and p < 0.01 confirms break detection
    assert p_val < 0.01
    assert f_stat > 10.0
