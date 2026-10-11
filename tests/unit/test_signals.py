"""
Unit tests for signals: spread construction, causal Z-score, envelopes, and state machine.
"""

import numpy as np
import pandas as pd

from statarb.config import SignalsConfig
from statarb.signals.envelopes import compute_bollinger_envelopes
from statarb.signals.generator import PositionState, SignalGenerator
from statarb.signals.zscore import compute_causal_zscore


def test_causal_zscore_excludes_current_observation():
    # Construct a series where S_t is an extreme outlier at t=50
    s_vals = np.zeros(100)
    s_vals[50] = 1000.0  # Massive outlier

    dates = pd.bdate_range("2022-01-01", periods=100)
    series = pd.Series(s_vals, index=dates)

    z_df = compute_causal_zscore(series, lookback=20)

    # At t=50, rolling_mean and rolling_std MUST NOT include the outlier at t=50!
    # They should reflect only t=30..49 (which are all zeros)
    assert z_df["rolling_mean"].iloc[50] == 0.0
    # The outlier at t=50 only enters into rolling statistics at t=51
    assert z_df["rolling_mean"].iloc[51] > 0.0


def test_bollinger_envelopes():
    dates = pd.bdate_range("2022-01-01", periods=50)
    mean = pd.Series(10.0, index=dates)
    std = pd.Series(2.0, index=dates)

    env = compute_bollinger_envelopes(mean, std, multiplier=2.0)
    assert np.all(env["upper_band"] == 14.0)
    assert np.all(env["lower_band"] == 6.0)
    assert np.all(env["equilibrium"] == 10.0)


def test_signal_generator_state_transitions():
    cfg = SignalsConfig(z_entry=2.0, z_exit=0.0, z_stop=3.5, max_holding_days=10)
    gen = SignalGenerator(cfg)

    # Construct sequence of Z-scores:
    # 0 -> 2.5 (enter short) -> 2.6 (stay short, no dup) -> 0.0 (exit MR)
    z_seq = [0.0, 2.5, 2.6, 1.0, 0.0, -2.5, -3.8]  # Last is stop loss
    dates = pd.bdate_range("2022-01-01", periods=len(z_seq))
    z_series = pd.Series(z_seq, index=dates)

    sig_df = gen.generate_signals(z_series)

    # Day 1: enter short
    assert sig_df["state"].iloc[1] == PositionState.SHORT_SPREAD.value
    assert sig_df["is_entry"].iloc[1]

    # Day 2: stay short (duplicate entry prevented)
    assert sig_df["state"].iloc[2] == PositionState.SHORT_SPREAD.value
    assert not sig_df["is_entry"].iloc[2]

    # Day 4: exit on mean reversion (Z = 0)
    assert sig_df["is_exit"].iloc[4]
    assert sig_df["exit_reason"].iloc[4] == "MEAN_REVERSION"

    # Day 5: enter long
    assert sig_df["state"].iloc[5] == PositionState.LONG_SPREAD.value

    # Day 6: stop loss trigger (Z = -3.8 < -3.5)
    assert sig_df["is_exit"].iloc[6]
    assert sig_df["exit_reason"].iloc[6] == "STOP_LOSS"
