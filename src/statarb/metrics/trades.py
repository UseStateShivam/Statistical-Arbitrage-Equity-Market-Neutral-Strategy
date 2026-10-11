"""
Trade-level analytics, win rate, holding durations, and exit breakdown.
"""

import pandas as pd


def compute_trade_metrics(trades_df: pd.DataFrame) -> dict[str, float]:
    """
    Computes summary metrics on completed trade records.
    """
    if trades_df.empty:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_holding_days": 0.0,
            "median_holding_days": 0.0,
            "mean_reversion_exits": 0,
            "stop_loss_exits": 0,
            "max_holding_exits": 0,
        }

    n_trades = len(trades_df)

    # Win rate: entry |Z| > exit |Z| indicates mean-reversion convergence
    if "entry_zscore" in trades_df.columns and "exit_zscore" in trades_df.columns:
        wins = (trades_df["entry_zscore"].abs() > trades_df["exit_zscore"].abs()).sum()
        win_rate = float(wins / n_trades)
    else:
        win_rate = 0.0

    avg_hold = (
        float(trades_df["holding_days"].mean()) if "holding_days" in trades_df.columns else 0.0
    )
    med_hold = (
        float(trades_df["holding_days"].median()) if "holding_days" in trades_df.columns else 0.0
    )

    mr_exits = (
        int((trades_df["exit_reason"] == "MEAN_REVERSION").sum())
        if "exit_reason" in trades_df.columns
        else 0
    )
    stop_exits = (
        int((trades_df["exit_reason"] == "STOP_LOSS").sum())
        if "exit_reason" in trades_df.columns
        else 0
    )
    time_exits = (
        int((trades_df["exit_reason"] == "MAX_HOLDING").sum())
        if "exit_reason" in trades_df.columns
        else 0
    )

    return {
        "total_trades": n_trades,
        "win_rate": win_rate,
        "avg_holding_days": avg_hold,
        "median_holding_days": med_hold,
        "mean_reversion_exits": mr_exits,
        "stop_loss_exits": stop_exits,
        "max_holding_exits": time_exits,
    }
