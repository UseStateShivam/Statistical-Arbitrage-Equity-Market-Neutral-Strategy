"""
Corporate actions and price adjustments handling (splits, dividends).
"""

import numpy as np
import pandas as pd


def compute_adjustment_factor(close: pd.Series, adj_close: pd.Series) -> pd.Series:
    """
    Computes adjustment ratio = adj_close / close.
    Handles zero or missing values safely.
    """
    valid_mask = (close > 0) & (~close.isna()) & (~adj_close.isna())
    factor = pd.Series(1.0, index=close.index, dtype=float)
    factor[valid_mask] = adj_close[valid_mask] / close[valid_mask]
    return factor


def apply_split_dividend_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies backward adjustments to open, high, low, and volume if only unadjusted columns exist.
    If 'adj_close' is present, scales open, high, and low by (adj_close / close).
    Scales volume inversely by the adjustment factor to preserve traded dollar volume.
    """
    df = df.copy()
    if "adj_close" not in df.columns or "close" not in df.columns:
        return df

    factor = compute_adjustment_factor(df["close"], df["adj_close"])

    if "adj_open" not in df.columns and "open" in df.columns:
        df["adj_open"] = df["open"] * factor
    if "adj_high" not in df.columns and "high" in df.columns:
        df["adj_high"] = df["high"] * factor
    if "adj_low" not in df.columns and "low" in df.columns:
        df["adj_low"] = df["low"] * factor
    if "adj_volume" not in df.columns and "volume" in df.columns:
        # Traded dollar volume = price * volume -> adj_vol = volume / factor
        safe_factor = np.where(factor > 0, factor, 1.0)
        df["adj_volume"] = df["volume"] / safe_factor

    return df
