"""
Command-Line Interface for the Statistical Arbitrage quantitative research platform.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from statarb.backtest.attribution import compute_cost_drag_attribution
from statarb.backtest.walk_forward import WalkForwardEngine
from statarb.config import AppConfig
from statarb.data.ingestion import IngestionManager
from statarb.data.universe import UniverseFilter
from statarb.logging_config import setup_logger
from statarb.metrics.performance import (
    compute_annualized_volatility,
    compute_cagr,
    compute_cumulative_return,
    compute_monthly_returns,
    compute_yearly_returns,
)
from statarb.metrics.risk import (
    compute_calmar_ratio,
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_var_cvar,
)
from statarb.metrics.trades import compute_trade_metrics
from statarb.metrics.turnover import compute_annualized_turnover
from statarb.reporting.charts import (
    plot_cost_sensitivity,
    plot_drawdowns,
    plot_equity_curve,
    plot_exposures,
    plot_monthly_heatmap,
    plot_rolling_sharpe,
    plot_spread_zscore,
)
from statarb.reporting.report import generate_research_report
from statarb.reporting.tables import format_performance_summary_table
from statarb.research.candidate_generation import CandidatePairGenerator
from statarb.research.selection import PairSelector
from statarb.utils.reproducibility import create_manifest, set_seed

logger = setup_logger("statarb.cli")


def cmd_validate_config(args: argparse.Namespace) -> int:
    """Validates configuration file against Pydantic schema."""
    try:
        cfg = AppConfig.from_yaml(args.config)
        logger.info(f"Configuration at '{args.config}' is VALID.")
        print(f"SUCCESS: Configuration '{cfg.experiment.id}' successfully validated.")
        return 0
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


def cmd_ingest(args: argparse.Namespace) -> int:
    """Ingests, validates, and normalizes market data."""
    try:
        cfg = AppConfig.from_yaml(args.config)
        set_seed(cfg.experiment.random_seed)
        mgr = IngestionManager(cfg)
        df, _report = mgr.run_ingestion()
        print(f"SUCCESS: Ingested {len(df):,} records for {df['symbol'].nunique()} symbols.")
        return 0
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


def cmd_discover_pairs(args: argparse.Namespace) -> int:
    """Discovers and filters cointegrated equity pairs."""
    try:
        cfg = AppConfig.from_yaml(args.config)
        set_seed(cfg.experiment.random_seed)
        prices_file = Path(cfg.data.processed_dir) / "prices.parquet"
        if not prices_file.exists():
            # Ingest first
            mgr = IngestionManager(cfg)
            df, _ = mgr.run_ingestion()
        else:
            df = pd.read_parquet(prices_file)

        price_col = "adj_close" if "adj_close" in df.columns else "close"
        pnl = df.pivot(index="date", columns="symbol", values=price_col).sort_index()

        # Slice to formation window
        formation_prices = pnl.tail(cfg.backtest.formation_window_days)
        formation_returns = formation_prices.pct_change().dropna()

        universe_filter = UniverseFilter(cfg.universe)
        cand_gen = CandidatePairGenerator(
            cfg.research.candidate_generation, universe_filter.sector_map
        )
        candidates = cand_gen.generate_candidates(formation_returns)

        selector = PairSelector(cfg.research)
        records = selector.evaluate_candidates(candidates, formation_prices)

        out_dir = Path(cfg.reporting.output_dir) / cfg.experiment.id
        out_dir.mkdir(parents=True, exist_ok=True)
        rec_df = pd.DataFrame([r.to_dict() for r in records])
        rec_df.to_csv(out_dir / "candidate_pairs.csv", index=False)
        selected_df = rec_df[rec_df["is_selected"]]
        selected_df.to_csv(out_dir / "selected_pairs.csv", index=False)

        print(
            f"SUCCESS: Evaluated {len(records)} pairs. Selected {len(selected_df)} qualifying pairs."
        )
        return 0
    except Exception as e:
        logger.error(f"Pair discovery failed: {e}")
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


def cmd_backtest(args: argparse.Namespace) -> int:
    """Executes chronological walk-forward backtest."""
    try:
        cfg = AppConfig.from_yaml(args.config)
        set_seed(cfg.experiment.random_seed)

        prices_file = Path(cfg.data.processed_dir) / "prices.parquet"
        if not prices_file.exists():
            mgr = IngestionManager(cfg)
            df, _ = mgr.run_ingestion()
        else:
            df = pd.read_parquet(prices_file)

        engine = WalkForwardEngine(cfg)
        wf_res = engine.run(df)

        out_dir = Path(cfg.reporting.output_dir) / cfg.experiment.id
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save artifacts
        if not wf_res.daily_portfolio.empty:
            wf_res.daily_portfolio.to_csv(out_dir / "daily_portfolio.csv")
            wf_res.daily_portfolio.to_parquet(out_dir / "daily_positions.parquet")
        if not wf_res.daily_exposures.empty:
            wf_res.daily_exposures.to_parquet(out_dir / "daily_exposures.parquet")
        if not wf_res.trades.empty:
            wf_res.trades.to_csv(out_dir / "trades.csv", index=False)

        print(f"SUCCESS: Walk-forward backtest completed. Sessions: {len(wf_res.daily_portfolio)}")
        return 0
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


def run_full_pipeline(cfg: AppConfig) -> int:
    """Executes the full end-to-end research and backtesting pipeline."""
    set_seed(cfg.experiment.random_seed)
    out_dir = Path(cfg.reporting.output_dir) / cfg.experiment.id
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingestion
    logger.info("Step 1: Data Ingestion & Quality Audit")
    mgr = IngestionManager(cfg)
    df, _quality_report = mgr.run_ingestion()

    # 2. Walk-Forward Backtesting
    logger.info("Step 2: Walk-Forward Econometric Simulation")
    engine = WalkForwardEngine(cfg)
    wf_res = engine.run(df)

    # 3. Performance Metrics Calculation
    logger.info("Step 3: Calculating Risk-Adjusted Metrics")
    daily_port = wf_res.daily_portfolio
    if daily_port.empty:
        logger.error("Daily portfolio output is empty.")
        return 1

    equity = daily_port["portfolio_equity"]
    returns = daily_port["daily_net_return"]

    cum_ret = compute_cumulative_return(equity)
    cagr = compute_cagr(equity)
    vol = compute_annualized_volatility(returns)
    sharpe = compute_sharpe_ratio(returns)
    sortino = compute_sortino_ratio(returns)
    max_dd, max_dd_days = compute_max_drawdown(equity)
    calmar = compute_calmar_ratio(cagr, max_dd)
    var_95, cvar_95 = compute_var_cvar(returns, 0.05)

    turnover_stats = compute_annualized_turnover(daily_port["daily_traded_notional"], equity)
    trade_stats = compute_trade_metrics(wf_res.trades)
    cost_attr = compute_cost_drag_attribution(daily_port)
    monthly_df = compute_monthly_returns(returns)
    yearly_ret = compute_yearly_returns(returns)

    metrics_dict = {
        "cumulative_return": cum_ret,
        "cagr": cagr,
        "annualized_volatility": vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_dd,
        "max_drawdown_days": max_dd_days,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "annualized_turnover": turnover_stats["annualized_two_way_turnover"],
        "total_trades": trade_stats["total_trades"],
        "win_rate": trade_stats["win_rate"],
        "avg_holding_days": trade_stats["avg_holding_days"],
        "total_transaction_costs": cost_attr.get("cumulative_transaction_costs", 0.0),
        "total_borrow_costs": cost_attr.get("cumulative_borrow_costs", 0.0),
    }

    # Save outputs
    daily_port.to_csv(out_dir / "daily_portfolio.csv")
    daily_port.to_parquet(out_dir / "daily_positions.parquet")
    daily_port.to_csv(out_dir / "equity_curve.csv")
    if not wf_res.daily_exposures.empty:
        wf_res.daily_exposures.to_parquet(out_dir / "daily_exposures.parquet")
    if not wf_res.trades.empty:
        wf_res.trades.to_csv(out_dir / "trades.csv", index=False)

    monthly_df.to_csv(out_dir / "monthly_returns.csv")
    yearly_ret.to_csv(out_dir / "yearly_metrics.csv")

    with open(out_dir / "performance_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)

    # 4. Cost Sensitivity Suite (0, 5, 10, 15, 20 bps)
    logger.info("Step 4: Running Execution Cost Sensitivity Analysis")
    sensitivity_rows = []
    base_costs = daily_port["daily_transaction_costs"]
    base_notional = daily_port["daily_traded_notional"]

    for slip_bps in [0.0, 5.0, 10.0, 15.0, 20.0]:
        slip_diff_rate = (slip_bps - cfg.execution.slippage_bps) / 10000.0
        adj_costs = base_costs + (base_notional * slip_diff_rate)
        adj_daily_net_pnl = (
            daily_port["daily_gross_pnl"] - adj_costs - daily_port["daily_borrow_costs"]
        )
        adj_equity = cfg.portfolio.initial_capital + adj_daily_net_pnl.cumsum()
        adj_returns = adj_daily_net_pnl / adj_equity.shift(1).fillna(cfg.portfolio.initial_capital)

        s_cagr = compute_cagr(adj_equity)
        s_vol = compute_annualized_volatility(adj_returns)
        s_sharpe = compute_sharpe_ratio(adj_returns)
        s_dd, _ = compute_max_drawdown(adj_equity)

        sensitivity_rows.append(
            {
                "slippage_bps": slip_bps,
                "cagr": s_cagr,
                "volatility": s_vol,
                "sharpe_ratio": s_sharpe,
                "max_drawdown": s_dd,
            }
        )

    cost_sens_df = pd.DataFrame(sensitivity_rows)
    cost_sens_df.to_csv(out_dir / "cost_sensitivity.csv", index=False)

    # 5. Visualizations
    logger.info("Step 5: Generating Publication-Quality Visualizations")
    plot_equity_curve(daily_port, figures_dir / "equity_curve.png", dpi=cfg.reporting.dpi)
    plot_drawdowns(daily_port, figures_dir / "drawdowns.png", dpi=cfg.reporting.dpi)
    plot_rolling_sharpe(daily_port, figures_dir / "rolling_sharpe.png", dpi=cfg.reporting.dpi)
    plot_monthly_heatmap(monthly_df, figures_dir / "monthly_heatmap.png", dpi=cfg.reporting.dpi)
    plot_cost_sensitivity(cost_sens_df, figures_dir / "cost_sensitivity.png", dpi=cfg.reporting.dpi)

    if not wf_res.daily_exposures.empty:
        plot_exposures(wf_res.daily_exposures, figures_dir / "exposures.png", dpi=cfg.reporting.dpi)

    # Plot sample pair signals
    for fold_res in wf_res.fold_results.values():
        if fold_res.pair_signals:
            pair_id, sig_df = next(iter(fold_res.pair_signals.items()))
            plot_spread_zscore(
                pair_id, sig_df, figures_dir / "sample_spread_zscore.png", dpi=cfg.reporting.dpi
            )
            break

    # 6. Generate Research Report
    logger.info("Step 6: Compiling Final Markdown Research Report")
    generate_research_report(
        experiment_id=cfg.experiment.id,
        metrics=metrics_dict,
        monthly_df=monthly_df,
        cost_attribution=cost_attr,
        cost_sensitivity_df=cost_sens_df,
        output_path=out_dir / "research_report.md",
    )

    # 7. Manifest
    manifest = create_manifest(
        experiment_id=cfg.experiment.id,
        config_dict=cfg.model_dump(),
        dataset_info={"rows": len(df), "symbols": int(df["symbol"].nunique())},
        success=True,
    )
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 70)
    print("STATISTICAL ARBITRAGE & EQUITY MARKET NEUTRAL STRATEGY RESULTS")
    print("=" * 70)
    print(format_performance_summary_table(metrics_dict))
    print(f"\nAll artifacts, reports, and figures persisted to:\n{out_dir.resolve()}")
    return 0


def cmd_run_all(args: argparse.Namespace) -> int:
    """Executes entire pipeline end-to-end."""
    try:
        cfg = AppConfig.from_yaml(args.config)
        return run_full_pipeline(cfg)
    except Exception as e:
        logger.error(f"Full pipeline run failed: {e}")
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="statarb", description="Production Quantitative Statistical Arbitrage & EMN Platform"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # validate-config
    p_val = subparsers.add_parser("validate-config", help="Validate YAML configuration")
    p_val.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    p_val.set_defaults(func=cmd_validate_config)

    # ingest
    p_ing = subparsers.add_parser("ingest", help="Ingest and validate market data")
    p_ing.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    p_ing.set_defaults(func=cmd_ingest)

    # discover-pairs
    p_disc = subparsers.add_parser("discover-pairs", help="Discover and screen cointegrated pairs")
    p_disc.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    p_disc.set_defaults(func=cmd_discover_pairs)

    # backtest
    p_bt = subparsers.add_parser("backtest", help="Execute walk-forward backtest")
    p_bt.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    p_bt.set_defaults(func=cmd_backtest)

    # run-all
    p_all = subparsers.add_parser("run-all", help="Execute full research and backtest pipeline")
    p_all.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    p_all.set_defaults(func=cmd_run_all)

    # report
    p_rep = subparsers.add_parser("report", help="Generate research report from experiment")
    p_rep.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    p_rep.set_defaults(func=cmd_run_all)

    args = parser.parse_args()
    code = args.func(args)
    sys.exit(code)


if __name__ == "__main__":
    main()
