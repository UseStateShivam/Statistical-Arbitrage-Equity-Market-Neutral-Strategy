"""
Unit tests for data normalization, validation, and universe filtering.
"""

from typing import cast

import pandas as pd

from statarb.data.normalization import normalize_ohlcv_dataframe
from statarb.data.universe import UniverseFilter
from statarb.data.validation import DataValidator


def test_normalize_ohlcv_dataframe(sample_ohlcv: pd.DataFrame):
    df_norm = normalize_ohlcv_dataframe(sample_ohlcv)
    assert "dollar_volume" in df_norm.columns
    assert "log_price" in df_norm.columns
    assert not df_norm.duplicated(subset=["date", "symbol"]).any()


def test_data_validator_detects_nonpositive_price():
    dates = pd.bdate_range("2022-01-01", periods=30)
    df = pd.DataFrame(
        {
            "date": dates,
            "symbol": "BAD",
            "close": [10.0] * 29 + [-5.0],  # Negative price
            "volume": [1000] * 30,
        }
    )
    validator = DataValidator()
    report = validator.validate_dataframe(df)
    assert report.valid_symbols_count == 0
    assert report.symbols["BAD"].non_positive_price_count == 1
    assert not report.symbols["BAD"].is_valid


def test_data_validator_detects_negative_volume():
    dates = pd.bdate_range("2022-01-01", periods=30)
    df = pd.DataFrame(
        {"date": dates, "symbol": "BADVOL", "close": [10.0] * 30, "volume": [1000] * 29 + [-100]}
    )
    validator = DataValidator()
    report = validator.validate_dataframe(df)
    assert report.symbols["BADVOL"].negative_volume_count == 1
    assert not report.symbols["BADVOL"].is_valid


def test_universe_filter_filters_penny_stocks(app_config):
    dates = pd.bdate_range("2022-01-01", periods=260)
    df = pd.DataFrame(
        {
            "date": dates,
            "symbol": "PENNY",
            "close": [2.50] * 260,  # Below $5 min
            "volume": [5_000_000] * 260,
        }
    )
    u_filter = UniverseFilter(app_config.universe)
    audit = u_filter.filter_universe(df, as_of_date=cast(pd.Timestamp, dates[-1]))
    assert len(audit) == 1
    assert not audit[0].is_eligible
    assert audit[0].rejection_reason is not None
    assert "Median price" in audit[0].rejection_reason
