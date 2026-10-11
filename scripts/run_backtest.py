#!/usr/bin/env python3
"""
Stand-alone script to run walk-forward backtest.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from statarb.backtest.walk_forward import WalkForwardEngine
from statarb.config import AppConfig
from statarb.logging_config import setup_logger

logger = setup_logger("scripts.run_backtest")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run walk-forward statistical arbitrage backtest")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--output-dir", default="reports/results", help="Output directory")
    args = parser.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    prices_path = Path(cfg.data.processed_dir) / "prices.parquet"

    if not prices_path.exists():
        logger.error(f"Processed prices not found at {prices_path}. Run ingestion first.")
        sys.exit(1)

    df = pd.read_parquet(prices_path)
    engine = WalkForwardEngine(cfg)
    wf_res = engine.run(df)

    out_dir = Path(args.output_dir) / cfg.experiment.id
    out_dir.mkdir(parents=True, exist_ok=True)

    if not wf_res.daily_portfolio.empty:
        wf_res.daily_portfolio.to_csv(out_dir / "daily_portfolio.csv")
    if not wf_res.trades.empty:
        wf_res.trades.to_csv(out_dir / "trades.csv", index=False)

    print(
        f"Backtest complete. Processed {len(wf_res.daily_portfolio)} out-of-sample trading dates."
    )
    print(f"Results persisted to {out_dir}")


if __name__ == "__main__":
    main()
