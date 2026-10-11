"""
Hedge ratio estimation and cointegrating regression models (OLS and TLS).
"""

from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm

from statarb.exceptions import CointegrationError


@dataclass
class HedgeRatioResult:
    alpha: float
    beta: float
    r_squared: float
    beta_std_error: float
    residuals: np.ndarray
    method: str
    n_obs: int


def estimate_ols_hedge_ratio(
    y: np.ndarray, x: np.ndarray, fit_intercept: bool = True
) -> HedgeRatioResult:
    """
    Estimates long-run cointegrating relationship:
    y = alpha + beta * x + epsilon
    where y = log(P_A) and x = log(P_B).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    valid_mask = (~np.isnan(y)) & (~np.isnan(x)) & (~np.isinf(y)) & (~np.isinf(x))
    y_clean = y[valid_mask]
    x_clean = x[valid_mask]

    n_obs = len(y_clean)
    if n_obs < 15:
        raise CointegrationError(f"Insufficient sample for hedge ratio estimation: {n_obs} < 15")

    if np.var(x_clean) < 1e-12 or np.var(y_clean) < 1e-12:
        raise CointegrationError("Degenerate regression: near-zero variance in asset prices.")

    if fit_intercept:
        X = sm.add_constant(x_clean)
        model = sm.OLS(y_clean, X).fit()
        alpha = float(model.params[0])
        beta = float(model.params[1])
        beta_se = float(model.bse[1])
    else:
        model = sm.OLS(y_clean, x_clean).fit()
        alpha = 0.0
        beta = float(model.params[0])
        beta_se = float(model.bse[0])

    r_squared = float(model.rsquared)
    residuals = np.asarray(model.resid, dtype=float)

    return HedgeRatioResult(
        alpha=alpha,
        beta=beta,
        r_squared=r_squared,
        beta_std_error=beta_se,
        residuals=residuals,
        method="OLS_WITH_INTERCEPT" if fit_intercept else "OLS_NO_INTERCEPT",
        n_obs=n_obs,
    )


def estimate_tls_hedge_ratio(y: np.ndarray, x: np.ndarray) -> HedgeRatioResult:
    """
    Total Least Squares (TLS) / Orthogonal Distance Regression.
    Treats both asset series symmetrically, removing dependent variable bias.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    valid_mask = (~np.isnan(y)) & (~np.isnan(x))
    y_clean = y[valid_mask]
    x_clean = x[valid_mask]

    n_obs = len(y_clean)
    if n_obs < 15:
        raise CointegrationError(f"Insufficient sample for TLS hedge ratio: {n_obs} < 15")

    x_mean = np.mean(x_clean)
    y_mean = np.mean(y_clean)

    # Demeaned data matrix
    Z = np.column_stack([x_clean - x_mean, y_clean - y_mean])
    # SVD decomposition
    _, _, vh = np.linalg.svd(Z)
    normal = vh[-1, :]  # eigenvector corresponding to smallest singular value

    if abs(normal[1]) < 1e-12:
        raise CointegrationError("Degenerate TLS normal vector.")

    beta = float(-normal[0] / normal[1])
    alpha = float(y_mean - beta * x_mean)
    residuals = y_clean - (alpha + beta * x_clean)

    # Approximate R-squared
    ss_tot = np.sum((y_clean - y_mean) ** 2)
    ss_res = np.sum(residuals**2)
    r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    return HedgeRatioResult(
        alpha=alpha,
        beta=beta,
        r_squared=max(0.0, r_squared),
        beta_std_error=0.0,
        residuals=residuals,
        method="TLS",
        n_obs=n_obs,
    )
