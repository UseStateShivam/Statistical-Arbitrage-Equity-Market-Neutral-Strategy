"""
Structural break, stability, and heteroskedasticity diagnostics.
"""

from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm
from scipy import stats


@dataclass
class StabilityResult:
    is_stable: bool
    chow_f_stat: float
    chow_pvalue: float
    hedge_ratio_cv: float
    variance_ratio: float
    warning_flag: str | None = None


def run_chow_test(
    y: np.ndarray, x: np.ndarray, split_idx: int | None = None
) -> tuple[float, float]:
    """
    Chow test for structural breakpoint in OLS cointegrating regression:
    y = alpha + beta * x + e.
    H0: Parameters are stable across both sub-periods.
    H1: Parameters broke at split_idx.
    """
    n = len(y)
    if split_idx is None:
        split_idx = n // 2

    k = 2  # constant + beta

    if split_idx < k + 5 or (n - split_idx) < k + 5:
        return 0.0, 1.0

    X = sm.add_constant(x)

    # 1. Pooled restricted regression
    m_pool = sm.OLS(y, X).fit()
    ssr_pool = np.sum(m_pool.resid**2)

    # 2. Sub-period 1
    m1 = sm.OLS(y[:split_idx], X[:split_idx]).fit()
    ssr1 = np.sum(m1.resid**2)

    # 3. Sub-period 2
    m2 = sm.OLS(y[split_idx:], X[split_idx:]).fit()
    ssr2 = np.sum(m2.resid**2)

    numerator = (ssr_pool - (ssr1 + ssr2)) / k
    denominator = (ssr1 + ssr2) / (n - 2 * k)

    if denominator <= 0:
        return 0.0, 1.0

    f_stat = numerator / denominator
    p_value = 1.0 - stats.f.cdf(f_stat, k, n - 2 * k)

    return float(f_stat), float(p_value)


def evaluate_pair_stability(
    y: np.ndarray,
    x: np.ndarray,
    subwindow_len: int = 63,
    max_cv_threshold: float = 0.25,
    chow_alpha: float = 0.01,
) -> StabilityResult:
    """
    Evaluates rolling hedge-ratio variation and structural break risk.
    """
    n = len(y)
    chow_f, chow_p = run_chow_test(y, x)

    # Rolling hedge ratios over subwindows
    rolling_betas = []
    step = max(5, subwindow_len // 4)
    for start in range(0, n - subwindow_len + 1, step):
        sub_x = x[start : start + subwindow_len]
        sub_y = y[start : start + subwindow_len]
        X = sm.add_constant(sub_x)
        try:
            m = sm.OLS(sub_y, X).fit()
            rolling_betas.append(m.params[1])
        except Exception:
            continue

    if len(rolling_betas) >= 3:
        mean_beta = np.mean(rolling_betas)
        std_beta = np.std(rolling_betas)
        cv = float(std_beta / abs(mean_beta)) if abs(mean_beta) > 1e-6 else 1.0
    else:
        cv = 0.0

    # Spread variance stability (ratio of second half variance to first half)
    half = n // 2
    var1 = np.var(y[:half] - x[:half])
    var2 = np.var(y[half:] - x[half:])
    var_ratio = float(var2 / var1) if var1 > 1e-12 else 1.0

    # Determine stability
    is_stable = True
    warning = None

    if chow_p < chow_alpha:
        is_stable = False
        warning = f"Chow test indicates structural break (p = {chow_p:.4f} < {chow_alpha})"
    elif cv > max_cv_threshold:
        is_stable = False
        warning = f"Hedge ratio CV ({cv:.2f}) exceeds threshold ({max_cv_threshold:.2f})"

    return StabilityResult(
        is_stable=is_stable,
        chow_f_stat=chow_f,
        chow_pvalue=chow_p,
        hedge_ratio_cv=cv,
        variance_ratio=var_ratio,
        warning_flag=warning,
    )
