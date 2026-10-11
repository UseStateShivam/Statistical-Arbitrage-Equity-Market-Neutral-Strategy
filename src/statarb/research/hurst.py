"""
Hurst exponent estimation via Variance-Time Scaling and Rescaled Range (R/S).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class HurstResult:
    hurst_exponent: float
    r_squared: float
    is_mean_reverting: bool  # H < 0.5
    is_valid: bool
    sample_size: int
    method: str


def estimate_hurst_variance_time(
    series: np.ndarray | pd.Series, min_lag: int = 2, max_lag: int | None = None
) -> HurstResult:
    """
    Estimates Hurst exponent H via Variance-Time scaling:
    Var(S_{t+tau} - S_t) ~ tau^{2H}
    Regressing ln(Var) on ln(tau) yields slope = 2H.
    - H < 0.5: Sub-diffusive / mean-reverting (OU process)
    - H = 0.5: Diffusive / Brownian motion (random walk)
    - H > 0.5: Super-diffusive / persistent (trending)
    """
    ts = np.asarray(series, dtype=float)
    ts = ts[~np.isnan(ts)]
    n = len(ts)

    if n < 30 or np.var(ts) < 1e-12:
        return HurstResult(
            hurst_exponent=0.5,
            r_squared=0.0,
            is_mean_reverting=False,
            is_valid=False,
            sample_size=n,
            method="variance_time",
        )

    if max_lag is None:
        max_lag = min(n // 4, 64)

    if max_lag <= min_lag:
        max_lag = min_lag + 8

    # Lags on log or power grid
    lags = [2, 4, 8, 16, 32, 64]
    lags = [lag for lag in lags if min_lag <= lag <= max_lag and lag < n]

    if len(lags) < 3:
        lags = [2, 4, 8, 12, 16]

    variances = []
    valid_lags = []

    for lag in lags:
        diffs = ts[lag:] - ts[:-lag]
        v = np.var(diffs, ddof=1)
        if v > 1e-12:
            variances.append(v)
            valid_lags.append(lag)

    if len(valid_lags) < 3:
        return HurstResult(
            hurst_exponent=0.5,
            r_squared=0.0,
            is_mean_reverting=False,
            is_valid=False,
            sample_size=n,
            method="variance_time",
        )

    log_lags = np.log(valid_lags)
    log_vars = np.log(variances)

    slope, intercept = np.polyfit(log_lags, log_vars, 1)
    hurst = float(slope / 2.0)

    # Calculate R-squared
    fitted = slope * log_lags + intercept
    ss_tot = np.sum((log_vars - np.mean(log_vars)) ** 2)
    ss_res = np.sum((log_vars - fitted) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    hurst_clamped = float(np.clip(hurst, 0.01, 0.99))
    is_valid = (0.01 < hurst < 0.99) and (r2 > 0.50)

    return HurstResult(
        hurst_exponent=hurst_clamped,
        r_squared=r2,
        is_mean_reverting=(hurst_clamped < 0.50),
        is_valid=is_valid,
        sample_size=n,
        method="variance_time",
    )


def estimate_hurst_rs(
    series: np.ndarray | pd.Series, min_lag: int = 10, max_lag: int | None = None
) -> HurstResult:
    """
    Estimates Hurst exponent H using Mandelbrot's Rescaled Range (R/S) analysis
    on the differenced stationary increments Delta S_t.
    """
    ts = np.asarray(series, dtype=float)
    ts = ts[~np.isnan(ts)]
    # Use increments to assess long memory
    inc = np.diff(ts)
    n = len(inc)

    if n < 30 or np.var(inc) < 1e-12:
        return HurstResult(
            hurst_exponent=0.5,
            r_squared=0.0,
            is_mean_reverting=False,
            is_valid=False,
            sample_size=n,
            method="rs",
        )

    if max_lag is None:
        max_lag = min(n // 2, 80)
    if max_lag <= min_lag:
        max_lag = min_lag + 10

    lags = [10, 20, 40, 60, 80]
    lags = [l for l in lags if l <= max_lag and l < n]
    if len(lags) < 3:
        lags = [min_lag, (min_lag + max_lag) // 2, max_lag]

    rs_values = []
    valid_lags = []

    for lag in lags:
        num_sub = n // lag
        if num_sub < 1:
            continue
        rs_sub = []
        for i in range(num_sub):
            sub = inc[i * lag : (i + 1) * lag]
            sub_mean = np.mean(sub)
            y = np.cumsum(sub - sub_mean)
            r = np.max(y) - np.min(y)
            s = np.std(sub, ddof=1)
            if s > 1e-10:
                rs_sub.append(r / s)
        if rs_sub:
            rs_values.append(np.mean(rs_sub))
            valid_lags.append(lag)

    if len(valid_lags) < 3:
        return estimate_hurst_variance_time(series, min_lag=2, max_lag=max_lag)

    log_lags = np.log(valid_lags)
    log_rs = np.log(rs_values)

    slope, intercept = np.polyfit(log_lags, log_rs, 1)
    hurst = float(slope)

    fitted = slope * log_lags + intercept
    ss_tot = np.sum((log_rs - np.mean(log_rs)) ** 2)
    ss_res = np.sum((log_rs - fitted) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    hurst_clamped = float(np.clip(hurst, 0.01, 0.99))
    is_valid = (0.01 < hurst < 0.99) and (r2 > 0.50)

    return HurstResult(
        hurst_exponent=hurst_clamped,
        r_squared=r2,
        is_mean_reverting=(hurst_clamped < 0.50),
        is_valid=is_valid,
        sample_size=n,
        method="rs",
    )


def estimate_hurst(series: np.ndarray | pd.Series, method: str = "variance_time") -> HurstResult:
    """Dispatches to configured Hurst estimation method."""
    if method == "variance_time":
        return estimate_hurst_variance_time(series)
    return estimate_hurst_rs(series)
