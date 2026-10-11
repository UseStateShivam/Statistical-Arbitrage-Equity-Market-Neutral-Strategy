#!/usr/bin/env python3
"""
Stand-alone script to download and validate market data.
"""

import argparse
import sys
from pathlib import Path

# Ensure src in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from statarb.config import AppConfig
from statarb.data.ingestion import IngestionManager
from statarb.logging_config import setup_logger

logger = setup_logger("scripts.download_data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and validate equity market data")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument(
        "--provider", choices=["synthetic", "parquet", "yahoo"], help="Override data provider"
    )
    args = parser.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    if args.provider:
        cfg.data.provider = args.provider

    logger.info(f"Executing data download with provider: '{cfg.data.provider}'")
    mgr = IngestionManager(cfg)
    df, report = mgr.run_ingestion()

    print(f"Ingested {len(df):,} price records across {df['symbol'].nunique()} equities.")
    print(
        f"Validation: {report.valid_symbols_count} valid, {report.excluded_symbols_count} excluded."
    )


if __name__ == "__main__":
    main()
