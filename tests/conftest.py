"""
Pytest configuration and shared test fixtures.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statarb.config import AppConfig
from statarb.data.providers import SyntheticDataProvider
from tests.fixtures.synthetic_processes import (
    generate_cointegrated_pair,
    generate_independent_random_walks,
    generate_stationary_ou_process,
)


@pytest.fixture
def app_config() -> AppConfig:
    """Provides validated test configuration."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    return AppConfig.from_yaml(config_path)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Provides small multi-symbol OHLCV dataset."""
    provider = SyntheticDataProvider(seed=42)
    df = provider.fetch_ohlcv(
        symbols=["AAPL", "MSFT", "GOOGL", "NVDA", "JPM", "BAC", "SPY"],
        start_date="2021-01-01",
        end_date="2022-12-31",
    )
    return df


@pytest.fixture
def stationary_spread() -> np.ndarray:
    return generate_stationary_ou_process(n=300, theta=0.08, sigma=0.015, seed=42)


@pytest.fixture
def cointegrated_prices() -> tuple:
    return generate_cointegrated_pair(n=400, alpha=0.3, beta=1.1, theta=0.07, seed=42)


@pytest.fixture
def random_walk_prices() -> tuple:
    return generate_independent_random_walks(n=400, seed=42)
