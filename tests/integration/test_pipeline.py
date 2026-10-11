"""
End-to-end integration test for the walk-forward statistical arbitrage engine.
"""

from statarb.backtest.accounting import reconcile_ledger
from statarb.backtest.walk_forward import WalkForwardEngine
from statarb.config import AppConfig
from statarb.data.providers import SyntheticDataProvider


def test_end_to_end_walk_forward_pipeline(app_config: AppConfig):
    # Setup fast test parameters
    app_config.backtest.formation_window_days = 150
    app_config.backtest.trading_window_days = 40
    app_config.backtest.rebalance_frequency_days = 40
    app_config.signals.spread_lookback = 30
    app_config.universe.min_history_days = 120

    provider = SyntheticDataProvider(seed=42)
    df = provider.fetch_ohlcv(
        symbols=["AAPL", "MSFT", "GOOGL", "NVDA", "JPM", "BAC", "SPY"],
        start_date="2021-01-01",
        end_date="2022-06-30",
    )

    engine = WalkForwardEngine(app_config)
    wf_res = engine.run(df)

    # Assertions
    assert len(wf_res.schedule) >= 1
    assert not wf_res.daily_portfolio.empty
    assert "portfolio_equity" in wf_res.daily_portfolio.columns

    # Verify reconciliation across all completed folds
    for fold_id, fold_res in wf_res.fold_results.items():
        if not fold_res.daily_portfolio.empty:
            is_rec, stats = reconcile_ledger(fold_res.ledger)
            assert is_rec, f"Fold {fold_id} ledger failed reconciliation: {stats}"
