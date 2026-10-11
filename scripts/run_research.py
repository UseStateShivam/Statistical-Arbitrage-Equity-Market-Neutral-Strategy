#!/usr/bin/env python3
"""
Stand-alone script to execute pair discovery, cointegration testing, and diagnostics.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from statarb.config import AppConfig
from statarb.data.universe import UniverseFilter
from statarb.logging_config import setup_logger
from statarb.research.candidate_generation import CandidatePairGenerator
from statarb.research.selection import PairSelector

logger = setup_logger("scripts.run_research")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run statistical arbitrage pair research and selection"
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--output-dir", default="reports/results", help="Output directory")
    args = parser.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    prices_path = Path(cfg.data.processed_dir) / "prices.parquet"

    if not prices_path.exists():
        logger.error(f"Processed prices not found at {prices_path}. Run ingestion first.")
        sys.exit(1)

    df = pd.read_parquet(prices_path)
    price_col = "adj_close" if "adj_close" in df.columns else "close"
    pnl = df.pivot(index="date", columns="symbol", values=price_col).sort_index()

    formation_prices = pnl.tail(cfg.backtest.formation_window_days)
    formation_returns = formation_prices.pct_change().dropna()

    universe_filter = UniverseFilter(cfg.universe)
    cand_gen = CandidatePairGenerator(cfg.research.candidate_generation, universe_filter.sector_map)
    candidates = cand_gen.generate_candidates(formation_returns)

    selector = PairSelector(cfg.research)
    records = selector.evaluate_candidates(candidates, formation_prices)

    out_dir = Path(args.output_dir) / cfg.experiment.id
    out_dir.mkdir(parents=True, exist_ok=True)

    rec_df = pd.DataFrame([r.to_dict() for r in records])
    rec_df.to_csv(out_dir / "candidate_pairs.csv", index=False)

    selected_df = rec_df[rec_df["is_selected"]]
    selected_df.to_csv(out_dir / "selected_pairs.csv", index=False)

    print(f"Research complete: {len(records)} pairs tested, {len(selected_df)} selected.")
    print(f"Artifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
