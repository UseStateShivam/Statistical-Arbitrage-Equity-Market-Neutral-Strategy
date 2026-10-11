"""
Deterministic synthetic statistical fixtures for econometric and unit testing.
"""

import numpy as np


def generate_stationary_ou_process(
    n: int = 500, theta: float = 0.08, sigma: float = 0.02, seed: int = 42
) -> np.ndarray:
    """Generates discrete Ornstein-Uhlenbeck stationary mean-reverting process."""
    rng = np.random.RandomState(seed)
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = s[t - 1] - theta * s[t - 1] + rng.normal(0, sigma)
    return s


def generate_independent_random_walks(
    n: int = 500, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generates two independent geometric Brownian motions (non-cointegrated)."""
    rng = np.random.RandomState(seed)
    rw1 = 50.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n)))
    rw2 = 100.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.018, n)))
    return rw1, rw2


def generate_cointegrated_pair(
    n: int = 500,
    alpha: float = 0.5,
    beta: float = 1.2,
    theta: float = 0.06,
    sigma_ou: float = 0.015,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates genuine cointegrated asset prices:
    log(P_A) = alpha + beta * log(P_B) + S_t
    where S_t is stationary OU.
    """
    rng = np.random.RandomState(seed)
    # Common stochastic trend
    log_pb = 4.0 + np.cumsum(rng.normal(0.0003, 0.012, n))
    pb = np.exp(log_pb)

    # Stationary spread
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = spread[t - 1] - theta * spread[t - 1] + rng.normal(0, sigma_ou)

    log_pa = alpha + beta * log_pb + spread
    pa = np.exp(log_pa)

    return pa, pb


def generate_correlated_noncointegrated_pair(
    n: int = 500, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates two series sharing a common factor but with non-stationary residual
    (high correlation, zero cointegration).
    """
    rng = np.random.RandomState(seed)
    common_factor = np.cumsum(rng.normal(0, 0.015, n))
    idio1 = np.cumsum(rng.normal(0, 0.01, n))
    idio2 = np.cumsum(rng.normal(0, 0.01, n))

    pa = 100.0 * np.exp(0.8 * common_factor + 0.6 * idio1)
    pb = 100.0 * np.exp(0.8 * common_factor + 0.6 * idio2)
    return pa, pb


def generate_process_with_break(
    n: int = 500, break_idx: int = 250, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generates cointegrated pair with structural shift in beta at break_idx."""
    rng = np.random.RandomState(seed)
    log_pb = 4.0 + np.cumsum(rng.normal(0, 0.012, n))
    pb = np.exp(log_pb)

    pa = np.zeros(n)
    # Sub-period 1: beta = 1.0
    pa[:break_idx] = np.exp(0.0 + 1.0 * log_pb[:break_idx] + rng.normal(0, 0.01, break_idx))
    # Sub-period 2: beta = 2.0 (structural break)
    pa[break_idx:] = np.exp(0.0 + 2.0 * log_pb[break_idx:] + rng.normal(0, 0.01, n - break_idx))

    return pa, pb
