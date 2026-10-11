"""
Risk-adjusted performance metrics (Sharpe, Sortino, Calmar, Drawdown, VaR, CVaR).
"""

import numpy as np
import pandas as pd


def compute_sharpe_ratio(
    returns: pd.Series, risk_free_rate_annual: float = 0.0, annualization_factor: int = 252
) -> float:
    """
    Computes annualized Sharpe ratio:
    Sharpe = sqrt(252) * (mean(r) - rf_daily) / std(r)
    """
    clean_r = returns.dropna()
    if len(clean_r) < 2:
        return 0.0

    std_dev = clean_r.std(ddof=1)
    if std_dev < 1e-8:
        return 0.0

    rf_daily = risk_free_rate_annual / annualization_factor
    excess_mean = clean_r.mean() - rf_daily
    return float(np.sqrt(annualization_factor) * (excess_mean / std_dev))


def compute_drawdown_series(equity_series: pd.Series) -> pd.DataFrame:
    """
    Computes running peak, drawdown series ($), and drawdown percentage (%).
    """
    peak = equity_series.cummax()
    drawdown_dollars = equity_series - peak
    drawdown_pct = drawdown_dollars / peak

    return pd.DataFrame(
        {
            "equity": equity_series,
            "peak": peak,
            "drawdown_dollars": drawdown_dollars,
            "drawdown_pct": drawdown_pct,
        },
        index=equity_series.index,
    )


def compute_max_drawdown(equity_series: pd.Series) -> tuple[float, int]:
    """
    Computes maximum percentage drawdown (reported as positive float, e.g. 0.08 for 8%)
    and maximum drawdown duration in trading sessions.
    """
    if len(equity_series) < 2:
        return 0.0, 0

    dd_df = compute_drawdown_series(equity_series)
    max_dd = float(abs(dd_df["drawdown_pct"].min()))

    # Compute max drawdown duration
    is_underwater = dd_df["drawdown_dollars"] < 0
    max_duration = 0
    curr_duration = 0
    for underwater in is_underwater:
        if underwater:
            curr_duration += 1
            max_duration = max(max_duration, curr_duration)
        else:
            curr_duration = 0

    return max_dd, max_duration


def compute_sortino_ratio(
    returns: pd.Series, risk_free_rate_annual: float = 0.0, annualization_factor: int = 252
) -> float:
    """
    Computes annualized Sortino ratio using downside standard deviation.
    """
    clean_r = returns.dropna()
    if len(clean_r) < 2:
        return 0.0

    rf_daily = risk_free_rate_annual / annualization_factor
    excess_r = clean_r - rf_daily
    downside = excess_r[excess_r < 0]

    if len(downside) < 2:
        return 0.0

    downside_std = np.sqrt(np.mean(downside**2))
    if downside_std < 1e-8:
        return 0.0

    return float(np.sqrt(annualization_factor) * (excess_r.mean() / downside_std))


def compute_calmar_ratio(cagr: float, max_drawdown: float) -> float:
    """Computes Calmar ratio: CAGR / Max Drawdown."""
    if max_drawdown <= 1e-6:
        return 0.0
    return float(cagr / max_drawdown)


def compute_var_cvar(returns: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    """
    Computes historical Value at Risk (VaR) and Conditional VaR (Expected Shortfall).
    Reported as positive losses (e.g. 0.015 = 1.5% daily loss).
    """
    clean_r = returns.dropna()
    if len(clean_r) < 10:
        return 0.0, 0.0

    var = float(-np.percentile(clean_r, alpha * 100))
    tail_losses = clean_r[clean_r <= -var]
    cvar = float(-tail_losses.mean()) if len(tail_losses) > 0 else var

    return var, cvar
