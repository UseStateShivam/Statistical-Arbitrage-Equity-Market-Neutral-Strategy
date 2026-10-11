"""
Returns, volatility, and multi-period performance metrics.
"""

import numpy as np
import pandas as pd


def compute_cumulative_return(equity_series: pd.Series) -> float:
    """Computes total net cumulative return: (E_end - E_start) / E_start."""
    if len(equity_series) < 2 or equity_series.iloc[0] <= 0:
        return 0.0
    return float((equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0])


def compute_cagr(equity_series: pd.Series, annualization_factor: int = 252) -> float:
    """Computes Compound Annual Growth Rate (CAGR)."""
    n_days = len(equity_series)
    if n_days < 2 or equity_series.iloc[0] <= 0:
        return 0.0

    total_return = float(equity_series.iloc[-1] / equity_series.iloc[0])
    years = n_days / annualization_factor
    if years <= 0 or total_return <= 0:
        return 0.0

    return float((total_return ** (1.0 / years)) - 1.0)


def compute_annualized_volatility(returns: pd.Series, annualization_factor: int = 252) -> float:
    """Computes annualized return volatility: std * sqrt(252)."""
    clean_ret = returns.dropna()
    if len(clean_ret) < 2:
        return 0.0
    return float(clean_ret.std(ddof=1) * np.sqrt(annualization_factor))


def compute_monthly_returns(daily_returns: pd.Series) -> pd.DataFrame:
    """
    Computes Year x Month return matrix.
    """
    if daily_returns.empty:
        return pd.DataFrame()

    df = daily_returns.to_frame(name="return").copy()
    df.index = pd.to_datetime(df.index)
    df["year"] = df.index.year
    df["month"] = df.index.month

    # Compounded monthly returns: prod(1 + r) - 1
    monthly = (
        df.groupby(["year", "month"])["return"].apply(lambda s: np.prod(1.0 + s) - 1.0).unstack()
    )
    # Format month names
    month_names = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }
    monthly = monthly.rename(columns=month_names)
    return monthly


def compute_yearly_returns(daily_returns: pd.Series) -> pd.Series:
    """Computes compounded calendar year returns."""
    if daily_returns.empty:
        return pd.Series(dtype=float)

    df = daily_returns.to_frame(name="return").copy()
    df.index = pd.to_datetime(df.index)
    df["year"] = df.index.year
    return df.groupby("year")["return"].apply(lambda s: float(np.prod(1.0 + s) - 1.0))
