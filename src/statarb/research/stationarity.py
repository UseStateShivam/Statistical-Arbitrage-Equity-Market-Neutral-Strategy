"""
Stationarity testing via Augmented Dickey-Fuller (ADF) procedure.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from statarb.exceptions import InsufficientDataError


@dataclass
class ADFResult:
    stat: float
    pvalue: float
    usedlag: int
    nobs: int
    critical_values: dict[str, float]
    is_stationary: bool
    deterministic_terms: str
    lag_selection: str


def run_adf_test(
    series: pd.Series | np.ndarray,
    regression: str = "c",
    autolag: str = "AIC",
    maxlag: int | None = 10,
    significance_level: float = 0.05,
) -> ADFResult:
    """
    Executes Augmented Dickey-Fuller unit-root test on a time series.
    Null Hypothesis H0: Unit root present (non-stationary).
    Alternative H1: Stationary process.
    """
    clean_series = np.asarray(series, dtype=float)
    clean_series = clean_series[~np.isnan(clean_series)]

    if len(clean_series) < 20:
        raise InsufficientDataError(f"Insufficient sample for ADF test: {len(clean_series)} < 20")

    # Constant or zero variance check
    if np.all(clean_series == clean_series[0]) or np.var(clean_series) < 1e-12:
        return ADFResult(
            stat=0.0,
            pvalue=1.0,
            usedlag=0,
            nobs=len(clean_series),
            critical_values={"1%": -3.45, "5%": -2.87, "10%": -2.57},
            is_stationary=False,
            deterministic_terms=regression,
            lag_selection=autolag,
        )

    res = adfuller(clean_series, regression=regression, autolag=autolag, maxlag=maxlag)

    stat = float(res[0])
    pvalue = float(res[1])
    usedlag = int(res[2])
    nobs = int(res[3])
    crit_vals = {str(k): float(v) for k, v in res[4].items()}

    # Strict rejection of null hypothesis
    is_stationary = pvalue < significance_level and stat < crit_vals.get(
        f"{int(significance_level * 100)}%", crit_vals.get("5%", -2.86)
    )

    return ADFResult(
        stat=stat,
        pvalue=pvalue,
        usedlag=usedlag,
        nobs=nobs,
        critical_values=crit_vals,
        is_stationary=is_stationary,
        deterministic_terms=regression,
        lag_selection=autolag,
    )
