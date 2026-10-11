"""
Half-life estimation via discrete-time Ornstein-Uhlenbeck process.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class HalfLifeResult:
    half_life_days: float
    reversion_speed: float  # -b
    intercept: float
    coefficient: float  # b
    coeff_pvalue: float
    r_squared: float
    is_valid_mean_reverting: bool
    rejection_reason: str | None = None


def estimate_half_life(
    spread: np.ndarray | pd.Series, min_half_life: float = 2.0, max_half_life: float = 63.0
) -> HalfLifeResult:
    """
    Fits discrete Ornstein-Uhlenbeck AR(1) representation:
    Delta S_t = a + b * S_{t-1} + u_t

    Half-life: t_{1/2} = -ln(2) / b, valid ONLY when b < 0.
    """
    s = np.asarray(spread, dtype=float)
    s = s[~np.isnan(s)]

    if len(s) < 20:
        return HalfLifeResult(
            half_life_days=np.nan,
            reversion_speed=0.0,
            intercept=0.0,
            coefficient=0.0,
            coeff_pvalue=1.0,
            r_squared=0.0,
            is_valid_mean_reverting=False,
            rejection_reason="Insufficient sample (< 20 observations)",
        )

    # Delta S_t = S_t - S_{t-1}
    delta_s = np.diff(s)
    lag_s = s[:-1]

    # Regress delta_s on lag_s with constant
    X = sm.add_constant(lag_s)
    try:
        model = sm.OLS(delta_s, X).fit()
    except Exception as e:
        return HalfLifeResult(
            half_life_days=np.nan,
            reversion_speed=0.0,
            intercept=0.0,
            coefficient=0.0,
            coeff_pvalue=1.0,
            r_squared=0.0,
            is_valid_mean_reverting=False,
            rejection_reason=f"OLS regression failed: {e}",
        )

    a = float(model.params[0])
    b = float(model.params[1])
    pvalue = float(model.pvalues[1])
    r2 = float(model.rsquared)

    # Check stationarity condition: b must be strictly negative
    if b >= 0.0:
        return HalfLifeResult(
            half_life_days=np.nan,
            reversion_speed=0.0,
            intercept=a,
            coefficient=b,
            coeff_pvalue=pvalue,
            r_squared=r2,
            is_valid_mean_reverting=False,
            rejection_reason=f"Non-mean-reverting coefficient b = {b:.4f} >= 0",
        )

    # Compute half-life
    half_life = float(-np.log(2.0) / b)
    reversion_speed = float(-b)

    # Check economically viable bounds
    rejection_reason = None
    if half_life < min_half_life:
        rejection_reason = (
            f"Half-life {half_life:.2f}d < min allowable {min_half_life}d (microstructure noise)"
        )
    elif half_life > max_half_life:
        rejection_reason = f"Half-life {half_life:.2f}d > max allowable {max_half_life}d (too slow)"

    is_valid = rejection_reason is None

    return HalfLifeResult(
        half_life_days=half_life,
        reversion_speed=reversion_speed,
        intercept=a,
        coefficient=b,
        coeff_pvalue=pvalue,
        r_squared=r2,
        is_valid_mean_reverting=is_valid,
        rejection_reason=rejection_reason,
    )
