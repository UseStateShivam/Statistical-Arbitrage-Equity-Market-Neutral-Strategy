"""
Chronological single-window backtest engine with causal execution timing.
"""

from dataclasses import dataclass

import pandas as pd

from statarb.config import AppConfig
from statarb.execution.fills import ExecutionSimulator, FillRecord
from statarb.execution.ledger import PortfolioLedger
from statarb.logging_config import setup_logger
from statarb.portfolio.construction import PortfolioConstructor
from statarb.portfolio.exposure import compute_portfolio_exposure
from statarb.research.selection import PairSelectionRecord
from statarb.signals.generator import SignalGenerator
from statarb.signals.spread import compute_spread
from statarb.signals.zscore import compute_causal_zscore

logger = setup_logger("statarb.backtest.engine")


@dataclass
class BacktestResult:
    daily_portfolio: pd.DataFrame
    daily_exposures: pd.DataFrame
    trades: pd.DataFrame
    fills: list[FillRecord]
    pair_signals: dict[str, pd.DataFrame]
    ledger: PortfolioLedger


class BacktestEngine:
    """
    Executes deterministic chronological backtest with zero look-ahead bias:
    Day t close signal -> Day t+1 open execution -> Day t+1 close mark-to-market.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.signal_gen = SignalGenerator(config.signals)
        self.constructor = PortfolioConstructor(config.portfolio)
        self.executor = ExecutionSimulator(config.execution)

    def run(
        self,
        selected_pairs: list[PairSelectionRecord],
        ohlcv_df: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        initial_capital: float | None = None,
    ) -> BacktestResult:
        """
        Runs day-by-day simulation across the specified trading window.
        """
        capital = initial_capital or self.config.portfolio.initial_capital
        ledger = PortfolioLedger(initial_capital=capital)

        # Filter price history up to end_date
        df = ohlcv_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["symbol", "date"])

        # Pivot prices
        price_col = "adj_close" if "adj_close" in df.columns else "close"
        close_panel = df.pivot(index="date", columns="symbol", values=price_col).sort_index()
        open_panel = df.pivot(index="date", columns="symbol", values="open").sort_index()
        vol_panel = df.pivot(index="date", columns="symbol", values="volume").sort_index()

        all_trading_dates = close_panel.index
        window_dates = [d for d in all_trading_dates if start_date <= d <= end_date]

        if len(window_dates) < 2:
            logger.warning(f"Insufficient trading dates in backtest window: {len(window_dates)}")
            return BacktestResult(
                daily_portfolio=pd.DataFrame(),
                daily_exposures=pd.DataFrame(),
                trades=pd.DataFrame(),
                fills=[],
                pair_signals={},
                ledger=ledger,
            )

        # Prepare pair metadata
        active_pair_dict = {r.pair_id: r for r in selected_pairs if r.is_selected}
        pair_pairs = {p_id: (r.asset_a, r.asset_b) for p_id, r in active_pair_dict.items()}
        pair_betas = {p_id: r.beta for p_id, r in active_pair_dict.items()}
        {p_id: r.alpha for p_id, r in active_pair_dict.items()}
        pair_stds = {p_id: r.spread_std for p_id, r in active_pair_dict.items()}

        pair_signals_history: dict[str, pd.DataFrame] = {}
        daily_exposures_records: list[dict] = []
        trade_records: list[dict] = []

        # Track active trade state per pair: pair_id -> {entry_date, entry_z, direction, notional}
        open_trades: dict[str, dict] = {}

        pending_target_dollars: dict[str, float] = {}

        for i, dt in enumerate(window_dates):
            # 1. MORNING OF DAY dt: Execute pending orders generated at previous day's close
            daily_traded_notional = 0.0
            daily_tx_costs = 0.0

            if pending_target_dollars and dt in open_panel.index:
                open_prices = open_panel.loc[dt].dropna().to_dict()
                adv_shares = vol_panel.loc[:dt].tail(20).mean().to_dict()

                fills = self.executor.execute_rebalance(
                    date=dt,
                    current_shares=ledger.shares,
                    target_dollar_allocations=pending_target_dollars,
                    open_prices=open_prices,
                    trailing_adv_shares=adv_shares,
                )

                if fills:
                    ledger.apply_fills(fills)
                    daily_traded_notional = sum(f.traded_notional for f in fills)
                    daily_tx_costs = sum(f.total_cost for f in fills)

            # 2. EVENING OF DAY dt: Mark to market at close prices
            close_prices = close_panel.loc[dt].dropna().to_dict()

            # Short borrow fee calculation
            short_mv = sum(
                sh * close_prices.get(sym, 0.0) for sym, sh in ledger.shares.items() if sh < 0
            )
            daily_borrow = self.executor.cost_model.compute_daily_borrow_cost(short_mv)
            daily_interest = self.executor.cost_model.compute_daily_cash_interest(ledger.cash)

            snapshot = ledger.mark_to_market(
                date=dt,
                close_prices=close_prices,
                daily_borrow_cost=daily_borrow,
                daily_cash_interest=daily_interest,
                traded_notional=daily_traded_notional,
                transaction_costs=daily_tx_costs,
            )

            # Record exposure metrics
            holdings_values = {
                sym: sh * close_prices.get(sym, 0.0) for sym, sh in ledger.shares.items()
            }
            exp_metrics = compute_portfolio_exposure(holdings_values, snapshot.portfolio_equity)
            daily_exposures_records.append(
                {
                    "date": dt,
                    "gross_exposure": exp_metrics.gross_exposure_pct,
                    "net_exposure": exp_metrics.net_exposure_pct,
                    "long_exposure": exp_metrics.long_exposure_pct,
                    "short_exposure": exp_metrics.short_exposure_pct,
                    "active_pairs": len(open_trades),
                }
            )

            # 3. END OF DAY dt: Compute signals on available data strictly up to dt close
            current_pair_directions: dict[str, int] = {}

            for pair_id, r in active_pair_dict.items():
                asset_a, asset_b = r.asset_a, r.asset_b
                if asset_a not in close_panel.columns or asset_b not in close_panel.columns:
                    continue

                # Historical series up to dt
                hist_p_a = close_panel.loc[:dt, asset_a].dropna()
                hist_p_b = close_panel.loc[:dt, asset_b].dropna()
                common_idx = hist_p_a.index.intersection(hist_p_b.index)

                if len(common_idx) < self.config.signals.spread_lookback:
                    continue

                p_a = hist_p_a.loc[common_idx]
                p_b = hist_p_b.loc[common_idx]

                # Compute spread
                spread = compute_spread(p_a, p_b, alpha=r.alpha, beta=r.beta)
                # Compute causal lagged Z-score
                z_df = compute_causal_zscore(spread, lookback=self.config.signals.spread_lookback)
                # Generate signal state
                sig_df = self.signal_gen.generate_signals(z_df["z_score"], spread)

                curr_sig = sig_df.iloc[-1]
                pair_signals_history[pair_id] = sig_df
                direction = int(curr_sig["direction"])
                current_pair_directions[pair_id] = direction

                # Trade tracking
                if curr_sig["is_entry"] and pair_id not in open_trades:
                    open_trades[pair_id] = {
                        "entry_date": dt,
                        "entry_z": curr_sig["z_score"],
                        "direction": "LONG_SPREAD" if direction == 1 else "SHORT_SPREAD",
                        "entry_spread": float(curr_sig.get("spread", 0.0)),
                    }
                elif curr_sig["is_exit"] and pair_id in open_trades:
                    tr = open_trades.pop(pair_id)
                    holding_len = (dt - tr["entry_date"]).days
                    trade_records.append(
                        {
                            "pair_id": pair_id,
                            "entry_date": tr["entry_date"],
                            "exit_date": dt,
                            "direction": tr["direction"],
                            "entry_zscore": tr["entry_z"],
                            "exit_zscore": curr_sig["z_score"],
                            "holding_days": holding_len,
                            "exit_reason": curr_sig["exit_reason"],
                        }
                    )

            # 4. Generate target positions for NEXT day's open
            pending_target_dollars = self.constructor.construct_target_positions(
                pair_directions=current_pair_directions,
                pair_betas=pair_betas,
                pair_pairs=pair_pairs,
                pair_spread_stds=pair_stds,
                portfolio_equity=snapshot.portfolio_equity,
            )

        # Finalize outputs
        daily_portfolio = ledger.to_dataframe()
        daily_exposures = (
            pd.DataFrame(daily_exposures_records).set_index("date")
            if daily_exposures_records
            else pd.DataFrame()
        )
        trades_df = pd.DataFrame(trade_records)

        return BacktestResult(
            daily_portfolio=daily_portfolio,
            daily_exposures=daily_exposures,
            trades=trades_df,
            fills=ledger.fills_history,
            pair_signals=pair_signals_history,
            ledger=ledger,
        )
