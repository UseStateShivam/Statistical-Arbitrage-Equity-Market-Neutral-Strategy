"""
Causal rolling Z-score computation excluding future information.
"""

import numpy as np
import pandas as pd


def compute_causal_zscore(
    spread: pd.Series, lookback: int = 63, min_periods: int | None = None
) -> pd.DataFrame:
    """
    Computes strictly lagged rolling Z-score:
    mu_{t-1} = Mean(S_{t-lookback} ... S_{t-1})
    sigma_{t-1} = Std(S_{t-lookback} ... S_{t-1})
    Z_t = (S_t - mu_{t-1}) / sigma_{t-1}

    Crucial: Day t observation S_t is NEVER included in mu_{t-1} or sigma_{t-1}.
    """
    if min_periods is None:
        min_periods = lookback // 2

    # Shift by 1 to guarantee strictly historical window ending at t-1
    lagged_spread = spread.shift(1)

    rolling_mean = lagged_spread.rolling(window=lookback, min_periods=min_periods).mean()
    rolling_std = lagged_spread.rolling(window=lookback, min_periods=min_periods).std(ddof=1)

    # Protect against division by zero or near-zero volatility
    safe_std = rolling_std.copy()
    safe_std[safe_std < 1e-6] = np.nan

    z_score = (spread - rolling_mean) / safe_std

    # Clean infinite values
    z_score = z_score.replace([np.inf, -np.inf], np.nan)

    res = pd.DataFrame(
        {
            "spread": spread,
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std,
            "z_score": z_score,
        },
        index=spread.index,
    )

    return res
