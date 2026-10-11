"""
Market data normalization and panel alignment utilities.
"""

import numpy as np
import pandas as pd

from statarb.exceptions import DataValidationError


def normalize_ohlcv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names, datetime types, and sorting order.
    Expected columns: date, symbol, open, high, low, close, volume, adj_close.
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    if "date" not in df.columns:
        raise DataValidationError("DataFrame missing required 'date' column.")
    if "symbol" not in df.columns:
        raise DataValidationError("DataFrame missing required 'symbol' column.")

    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()

    # Drop duplicate date-symbol records, keeping first
    df = (
        df.drop_duplicates(subset=["date", "symbol"])
        .sort_values(by=["symbol", "date"])
        .reset_index(drop=True)
    )

    # Calculate dollar volume if not present
    if "dollar_volume" not in df.columns:
        price_col = "adj_close" if "adj_close" in df.columns else "close"
        if price_col in df.columns and "volume" in df.columns:
            df["dollar_volume"] = df[price_col] * df["volume"]

    # Calculate log prices
    price_col = "adj_close" if "adj_close" in df.columns else "close"
    if price_col in df.columns:
        valid_prices = df[price_col] > 0
        df["log_price"] = np.nan
        df.loc[valid_prices, "log_price"] = np.log(df.loc[valid_prices, price_col])

    return df


def pivot_price_panel(df: pd.DataFrame, value_col: str = "adj_close") -> pd.DataFrame:
    """
    Pivots long-format OHLCV dataframe into a wide (dates x symbols) price panel.
    """
    if "date" not in df.columns or "symbol" not in df.columns or value_col not in df.columns:
        raise DataValidationError(
            f"Required columns missing for pivot (need date, symbol, {value_col})."
        )

    panel = df.pivot(index="date", columns="symbol", values=value_col)
    panel = panel.sort_index()
    return panel
