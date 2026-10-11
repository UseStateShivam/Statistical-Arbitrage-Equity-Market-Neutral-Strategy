"""
Spread series computation and transformation.
"""

import numpy as np
import pandas as pd


def compute_spread(price_a: pd.Series, price_b: pd.Series, alpha: float, beta: float) -> pd.Series:
    """
    Computes static-hedge candidate spread:
    S_t = log(P_A(t)) - alpha - beta * log(P_B(t))
    """
    valid_mask = (price_a > 0) & (price_b > 0) & (~price_a.isna()) & (~price_b.isna())
    spread = pd.Series(np.nan, index=price_a.index, dtype=float)

    log_a = np.log(price_a[valid_mask])
    log_b = np.log(price_b[valid_mask])

    spread[valid_mask] = log_a - alpha - (beta * log_b)
    return spread


def compute_dynamic_spread(
    price_a: pd.Series, price_b: pd.Series, alphas: pd.Series, betas: pd.Series
) -> pd.Series:
    """
    Computes spread using time-varying, strictly lagged hedge ratios:
    S_t = log(P_A(t)) - alpha_t - beta_t * log(P_B(t))
    """
    valid_mask = (price_a > 0) & (price_b > 0) & (~alphas.isna()) & (~betas.isna())
    spread = pd.Series(np.nan, index=price_a.index, dtype=float)

    log_a = np.log(price_a[valid_mask])
    log_b = np.log(price_b[valid_mask])

    spread[valid_mask] = log_a - alphas[valid_mask] - (betas[valid_mask] * log_b)
    return spread
