"""
Cointegration testing: Engle-Granger two-step with MacKinnon critical values,
Johansen VECM maximum likelihood, and Benjamini-Hochberg FDR control.
"""

from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from statarb.exceptions import CointegrationError, InsufficientDataError
from statarb.research.hedge_ratio import estimate_ols_hedge_ratio


@dataclass
class EngleGrangerResult:
    coint_stat: float
    pvalue: float
    crit_values: dict[str, float]
    alpha: float
    beta: float
    residuals: np.ndarray
    is_cointegrated: bool
    nobs: int


@dataclass
class JohansenResult:
    trace_stat_r0: float
    trace_crit95_r0: float
    trace_stat_r1: float
    trace_crit95_r1: float
    max_eig_stat_r0: float
    max_eig_crit95_r0: float
    max_eig_stat_r1: float
    max_eig_crit95_r1: float
    cointegration_rank: int
    is_cointegrated: bool


def run_engle_granger(
    y: np.ndarray, x: np.ndarray, significance_level: float = 0.05, trend: str = "c"
) -> EngleGrangerResult:
    """
    Executes Engle-Granger two-step cointegration test using MacKinnon (1990, 2010)
    response surface critical values.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    mask = (~np.isnan(y)) & (~np.isnan(x)) & (~np.isinf(y)) & (~np.isinf(x))
    y_clean = y[mask]
    x_clean = x[mask]

    nobs = len(y_clean)
    if nobs < 25:
        raise InsufficientDataError(f"Insufficient sample for Engle-Granger test: {nobs} < 25")

    try:
        coint_t, pvalue, crit_val = coint(y_clean, x_clean, trend=trend, autolag="AIC")
    except Exception as e:
        raise CointegrationError(f"Engle-Granger test failed: {e}") from e

    crit_dict = {"1%": float(crit_val[0]), "5%": float(crit_val[1]), "10%": float(crit_val[2])}

    # Estimate long-run parameters
    hedge_res = estimate_ols_hedge_ratio(y_clean, x_clean, fit_intercept=True)

    is_coint = (pvalue < significance_level) and (coint_t < crit_dict["5%"])

    return EngleGrangerResult(
        coint_stat=float(coint_t),
        pvalue=float(pvalue),
        crit_values=crit_dict,
        alpha=hedge_res.alpha,
        beta=hedge_res.beta,
        residuals=hedge_res.residuals,
        is_cointegrated=is_coint,
        nobs=nobs,
    )


def run_johansen(
    y: np.ndarray, x: np.ndarray, det_order: int = 0, k_ar_diff: int = 1
) -> JohansenResult:
    """
    Executes Johansen Maximum Likelihood cointegration test on bivariate [y, x] system.
    det_order: -1 (no constant), 0 (constant within cointegrating relation), 1 (linear trend).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    mask = (~np.isnan(y)) & (~np.isnan(x))
    data = np.column_stack([y[mask], x[mask]])

    if len(data) < 30:
        raise InsufficientDataError(
            "Insufficient sample for Johansen test: need >= 30 observations."
        )

    try:
        res = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
    except Exception as e:
        raise CointegrationError(f"Johansen test failed: {e}") from e

    # res.lr1: trace statistics [r=0, r=1]
    # res.cvt: critical values table (90%, 95%, 99%)
    # res.lr2: max eigenvalue statistics
    # res.cvm: critical values table for max eigenvalue
    trace_r0 = float(res.lr1[0])
    trace_crit95_r0 = float(res.cvt[0, 1])
    trace_r1 = float(res.lr1[1])
    trace_crit95_r1 = float(res.cvt[1, 1])

    max_eig_r0 = float(res.lr2[0])
    max_eig_crit95_r0 = float(res.cvm[0, 1])
    max_eig_r1 = float(res.lr2[1])
    max_eig_crit95_r1 = float(res.cvm[1, 1])

    # Rank selection:
    # If reject r=0 (trace > crit) and cannot reject r=1 (trace <= crit) -> rank = 1
    if trace_r0 > trace_crit95_r0:
        if trace_r1 > trace_crit95_r1:
            rank = 2  # Both stationary (degenerate)
        else:
            rank = 1  # Exactly 1 cointegrating relationship
    else:
        rank = 0  # No cointegration

    is_coint = rank == 1

    return JohansenResult(
        trace_stat_r0=trace_r0,
        trace_crit95_r0=trace_crit95_r0,
        trace_stat_r1=trace_r1,
        trace_crit95_r1=trace_crit95_r1,
        max_eig_stat_r0=max_eig_r0,
        max_eig_crit95_r0=max_eig_crit95_r0,
        max_eig_stat_r1=max_eig_r1,
        max_eig_crit95_r1=max_eig_crit95_r1,
        cointegration_rank=rank,
        is_cointegrated=is_coint,
    )


def apply_benjamini_hochberg(
    pvalues: list[float] | np.ndarray, alpha: float = 0.10
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies Benjamini-Hochberg (1995) procedure to control False Discovery Rate (FDR).
    Returns (rejected_boolean_array, adjusted_qvalues_array).
    """
    pvals = np.asarray(pvalues, dtype=float)
    m = len(pvals)
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    # Sort p-values while preserving original indices
    sorted_indices = np.argsort(pvals)
    sorted_pvals = pvals[sorted_indices]

    # Calculate BH adjusted q-values: q_i = min_{j >= i} (m * p_j / j)
    qvalues = np.zeros(m, dtype=float)
    ranks = np.arange(1, m + 1)
    raw_q = sorted_pvals * m / ranks

    # Enforce monotonicity backwards
    min_q = 1.0
    for i in range(m - 1, -1, -1):
        min_q = min(min_q, raw_q[i])
        qvalues[i] = min(min_q, 1.0)

    # Restore original ordering
    adjusted_q = np.zeros(m, dtype=float)
    adjusted_q[sorted_indices] = qvalues

    rejected = adjusted_q <= alpha
    return rejected, adjusted_q
