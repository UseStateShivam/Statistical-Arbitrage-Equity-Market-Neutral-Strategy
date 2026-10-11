"""
Volatility-adjusted Bollinger-style dynamic envelopes.
"""

import pandas as pd


def compute_bollinger_envelopes(
    rolling_mean: pd.Series,
    rolling_std: pd.Series,
    multiplier: float = 2.0,
    vol_scale: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Computes dynamic envelopes around the spread equilibrium:
    Upper = mu_{t-1} + k * sigma_{t-1} * vol_scale
    Lower = mu_{t-1} - k * sigma_{t-1} * vol_scale
    """
    if vol_scale is not None:
        effective_width = multiplier * rolling_std * vol_scale
    else:
        effective_width = multiplier * rolling_std

    upper_band = rolling_mean + effective_width
    lower_band = rolling_mean - effective_width

    return pd.DataFrame(
        {"upper_band": upper_band, "lower_band": lower_band, "equilibrium": rolling_mean},
        index=rolling_mean.index,
    )
