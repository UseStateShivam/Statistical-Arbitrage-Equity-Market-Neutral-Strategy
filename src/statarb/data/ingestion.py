"""
Data ingestion manager orchestrating caching, validation, and storage.
"""

import json
from pathlib import Path

import pandas as pd

from statarb.config import AppConfig
from statarb.data.corporate_actions import apply_split_dividend_adjustments
from statarb.data.normalization import normalize_ohlcv_dataframe
from statarb.data.providers import (
    BaseDataProvider,
    LocalParquetProvider,
    SyntheticDataProvider,
    YahooFinanceProvider,
)
from statarb.data.validation import DataQualityReport, DataValidator
from statarb.logging_config import setup_logger

logger = setup_logger("statarb.data.ingestion")


class IngestionManager:
    """Manages acquisition, validation, and caching of market data."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.raw_dir = Path(config.data.raw_dir)
        self.processed_dir = Path(config.data.processed_dir)
        self.sample_dir = Path(config.data.sample_dir)

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)

    def get_provider(self) -> BaseDataProvider:
        provider_type = self.config.data.provider
        if provider_type == "synthetic":
            return SyntheticDataProvider(seed=self.config.experiment.random_seed)
        elif provider_type == "parquet":
            return LocalParquetProvider(self.processed_dir)
        elif provider_type == "yahoo":
            return YahooFinanceProvider(cache_dir=self.raw_dir)
        else:
            raise ValueError(f"Unknown data provider type: {provider_type}")

    def run_ingestion(self) -> tuple[pd.DataFrame, DataQualityReport]:
        """
        Executes full ingestion workflow: provider fetch -> adjustments -> validation -> parquet cache.
        """
        logger.info(f"Initiating data ingestion using provider: '{self.config.data.provider}'")
        provider = self.get_provider()

        df = provider.fetch_ohlcv(
            symbols=self.config.universe.symbols,
            start_date=self.config.data.start_date,
            end_date=self.config.data.end_date,
        )

        logger.info(f"Retrieved {len(df):,} price records across {df['symbol'].nunique()} symbols.")

        # Corporate actions adjustment
        df = apply_split_dividend_adjustments(df)
        df = normalize_ohlcv_dataframe(df)

        # Validation
        validator = DataValidator(max_missing_pct=self.config.universe.max_missing_pct)
        quality_report = validator.validate_dataframe(df)
        logger.info(
            f"Validation complete: {quality_report.valid_symbols_count} valid symbols, "
            f"{quality_report.excluded_symbols_count} excluded."
        )

        # Save processed Parquet
        processed_file = self.processed_dir / "prices.parquet"
        df.to_parquet(processed_file, index=False)
        logger.info(f"Persisted processed dataset to: {processed_file}")

        # If synthetic or sample run, also save to sample directory
        sample_file = self.sample_dir / "sample_prices.parquet"
        df.to_parquet(sample_file, index=False)

        # Save validation report
        report_file = self.processed_dir / "data_quality_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(quality_report.to_dict(), f, indent=2)

        return df, quality_report
