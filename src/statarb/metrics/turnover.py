"""
Turnover computation and portfolio velocity metrics.
"""

import pandas as pd


def compute_annualized_turnover(
    daily_traded_notional: pd.Series, portfolio_equity: pd.Series, annualization_factor: int = 252
) -> dict[str, float]:
    """
    Computes annualized portfolio turnover:
    Annualized Turnover = 252 * mean_t(Daily Traded Notional_t / Portfolio Equity_t)
    Reports both two-way (total buys + sells) and one-way turnover.
    """
    valid_mask = (
        (portfolio_equity > 0) & (~portfolio_equity.isna()) & (~daily_traded_notional.isna())
    )
    daily_ratios = daily_traded_notional[valid_mask] / portfolio_equity[valid_mask]

    if daily_ratios.empty:
        return {
            "daily_turnover_mean": 0.0,
            "annualized_two_way_turnover": 0.0,
            "annualized_one_way_turnover": 0.0,
        }

    daily_mean = float(daily_ratios.mean())
    two_way_annual = float(daily_mean * annualization_factor)
    one_way_annual = float(two_way_annual / 2.0)

    return {
        "daily_turnover_mean": daily_mean,
        "annualized_two_way_turnover": two_way_annual,
        "annualized_one_way_turnover": one_way_annual,
    }
