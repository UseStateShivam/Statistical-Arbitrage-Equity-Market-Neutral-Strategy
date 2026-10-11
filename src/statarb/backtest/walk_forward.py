"""
Walk-forward chronological simulation engine.
Re-evaluates universe, cointegration, and pair selection at periodic boundaries
without look-ahead bias, stitching out-of-sample periods into a unified track record.
"""

from dataclasses import dataclass

import pandas as pd

from statarb.backtest.engine import BacktestEngine, BacktestResult
from statarb.config import AppConfig
from statarb.data.universe import UniverseFilter
from statarb.logging_config import setup_logger
from statarb.research.candidate_generation import CandidatePairGenerator
from statarb.research.selection import PairSelectionRecord, PairSelector
from statarb.utils.dates import WalkForwardWindow, generate_walk_forward_schedule

logger = setup_logger("statarb.backtest.walk_forward")


@dataclass
class WalkForwardResult:
    daily_portfolio: pd.DataFrame
    daily_exposures: pd.DataFrame
    trades: pd.DataFrame
    fills: list
    selected_pairs_history: dict[int, list[PairSelectionRecord]]
    fold_results: dict[int, BacktestResult]
    schedule: list[WalkForwardWindow]


class WalkForwardEngine:
    """
    Orchestrates periodic walk-forward re-estimation and out-of-sample evaluation.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.universe_filter = UniverseFilter(config.universe)
        self.candidate_gen = CandidatePairGenerator(
            config.research.candidate_generation, sector_map=self.universe_filter.sector_map
        )
        self.selector = PairSelector(config.research)
        self.backtest_engine = BacktestEngine(config)

    def run(self, ohlcv_df: pd.DataFrame) -> WalkForwardResult:
        """
        Executes rolling walk-forward backtest across all chronological folds.
        """
        df = ohlcv_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        price_col = "adj_close" if "adj_close" in df.columns else "close"
        close_panel = df.pivot(index="date", columns="symbol", values=price_col).sort_index()

        schedule = generate_walk_forward_schedule(
            trading_dates=close_panel.index,
            formation_days=self.config.backtest.formation_window_days,
            trading_days=self.config.backtest.trading_window_days,
            step_days=self.config.backtest.rebalance_frequency_days,
        )

        logger.info(f"Generated {len(schedule)} walk-forward chronological windows.")

        all_daily_portfolios: list[pd.DataFrame] = []
        all_daily_exposures: list[pd.DataFrame] = []
        all_trades: list[pd.DataFrame] = []
        all_fills: list = []
        selected_pairs_history: dict[int, list[PairSelectionRecord]] = {}
        fold_results: dict[int, BacktestResult] = {}

        current_capital = self.config.portfolio.initial_capital

        for window in schedule:
            logger.info(
                f"--- Fold {window.window_id}: Formation [{window.formation_start.strftime('%Y-%m-%d')} "
                f"to {window.formation_end.strftime('%Y-%m-%d')}] -> "
                f"Trading [{window.trading_start.strftime('%Y-%m-%d')} to {window.trading_end.strftime('%Y-%m-%d')}] ---"
            )

            # 1. Point-in-Time Universe Filter as of formation_end
            audit_records = self.universe_filter.filter_universe(
                df=df,
                as_of_date=window.formation_end,
                lookback_days=self.config.backtest.formation_window_days,
            )
            eligible_symbols = self.universe_filter.get_eligible_symbols(audit_records)
            logger.info(f"Fold {window.window_id}: {len(eligible_symbols)} eligible equities.")

            if len(eligible_symbols) < 2:
                logger.warning(
                    f"Fold {window.window_id}: Insufficient eligible equities; skipping."
                )
                continue

            # 2. Slice formation data strictly within [formation_start, formation_end]
            formation_prices = close_panel.loc[
                window.formation_start : window.formation_end, eligible_symbols
            ]
            formation_returns = formation_prices.pct_change().dropna()

            # 3. Candidate Generation
            candidates = self.candidate_gen.generate_candidates(formation_returns)
            logger.info(f"Fold {window.window_id}: Screened {len(candidates)} candidate pairs.")

            # 4. Cointegration & Multi-Criteria Selection
            selection_records = self.selector.evaluate_candidates(candidates, formation_prices)
            selected_pairs_history[window.window_id] = selection_records

            chosen_pairs = [r for r in selection_records if r.is_selected]
            # Cap to max active pairs
            chosen_pairs = chosen_pairs[: self.config.portfolio.max_active_pairs]
            logger.info(f"Fold {window.window_id}: Selected {len(chosen_pairs)} qualifying pairs.")

            # 5. Out-of-Sample Forward Backtest for this fold
            # Data available to engine is up to trading_end (engine enforces causal t close -> t+1 open)
            fold_df = df[df["date"] <= window.trading_end]

            fold_res = self.backtest_engine.run(
                selected_pairs=chosen_pairs,
                ohlcv_df=fold_df,
                start_date=window.trading_start,
                end_date=window.trading_end,
                initial_capital=current_capital,
            )

            fold_results[window.window_id] = fold_res

            if not fold_res.daily_portfolio.empty:
                all_daily_portfolios.append(fold_res.daily_portfolio)
                current_capital = float(fold_res.daily_portfolio["portfolio_equity"].iloc[-1])
            if not fold_res.daily_exposures.empty:
                all_daily_exposures.append(fold_res.daily_exposures)
            if not fold_res.trades.empty:
                all_trades.append(fold_res.trades)
            if fold_res.fills:
                all_fills.extend(fold_res.fills)

        # Concatenate out-of-sample series
        if all_daily_portfolios:
            unified_portfolio = pd.concat(all_daily_portfolios).sort_index()
            # Drop any duplicate boundary dates
            unified_portfolio = unified_portfolio[~unified_portfolio.index.duplicated(keep="last")]
        else:
            unified_portfolio = pd.DataFrame()

        if all_daily_exposures:
            unified_exposures = pd.concat(all_daily_exposures).sort_index()
            unified_exposures = unified_exposures[~unified_exposures.index.duplicated(keep="last")]
        else:
            unified_exposures = pd.DataFrame()

        unified_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

        return WalkForwardResult(
            daily_portfolio=unified_portfolio,
            daily_exposures=unified_exposures,
            trades=unified_trades,
            fills=all_fills,
            selected_pairs_history=selected_pairs_history,
            fold_results=fold_results,
            schedule=schedule,
        )
