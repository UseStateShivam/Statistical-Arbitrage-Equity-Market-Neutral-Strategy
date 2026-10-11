"""
Regression tests verifying strict absence of look-ahead bias and data leakage.
"""

import numpy as np
import pandas as pd

from statarb.signals.generator import SignalGenerator
from statarb.signals.spread import compute_spread
from statarb.signals.zscore import compute_causal_zscore


def test_future_price_perturbation_does_not_alter_past_signals(app_config):
    """
    Look-ahead regression test:
    Modifying future prices at t >= 50 MUST produce 100% IDENTICAL signals at t < 50.
    """
    n = 100
    dates = pd.bdate_range("2022-01-01", periods=n)
    np.random.seed(42)

    pa_base = pd.Series(100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, n))), index=dates)
    pb_base = pd.Series(50.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, n))), index=dates)

    # Base pipeline
    spread_base = compute_spread(pa_base, pb_base, alpha=0.0, beta=1.0)
    z_base = compute_causal_zscore(spread_base, lookback=20)
    gen = SignalGenerator(app_config.signals)
    sig_base = gen.generate_signals(z_base["z_score"])

    # Perturbed series: inject extreme price shocks only in the future (t >= 50)
    pa_perturbed = pa_base.copy()
    pb_perturbed = pb_base.copy()
    pa_perturbed.iloc[50:] = pa_perturbed.iloc[50:] * 5.0  # 500% future shock
    pb_perturbed.iloc[50:] = pb_perturbed.iloc[50:] * 0.1  # 90% future drop

    spread_pert = compute_spread(pa_perturbed, pb_perturbed, alpha=0.0, beta=1.0)
    z_pert = compute_causal_zscore(spread_pert, lookback=20)
    sig_pert = gen.generate_signals(z_pert["z_score"])

    # INVARIANT: Signals and Z-scores up to t=49 MUST be identical!
    pd.testing.assert_series_equal(
        z_base["z_score"].iloc[:50], z_pert["z_score"].iloc[:50], check_exact=True
    )
    pd.testing.assert_series_equal(
        sig_base["direction"].iloc[:50], sig_pert["direction"].iloc[:50], check_exact=True
    )
    pd.testing.assert_series_equal(
        sig_base["state"].iloc[:50], sig_pert["state"].iloc[:50], check_exact=True
    )


def test_rolling_window_strictly_lagged():
    """
    Verifies that day t rolling mean and std exclude day t spread observation.
    """
    dates = pd.bdate_range("2022-01-01", periods=40)
    # Zero spread everywhere except extreme spike at t=25
    spread = pd.Series(0.0, index=dates)
    spread.iloc[25] = 9999.0

    z_df = compute_causal_zscore(spread, lookback=15)

    # At t=25, mean must be 0.0, not contaminated by 9999.0
    assert z_df["rolling_mean"].iloc[25] == 0.0
    # Mean only reflects the spike at t=26
    assert z_df["rolling_mean"].iloc[26] > 0.0
