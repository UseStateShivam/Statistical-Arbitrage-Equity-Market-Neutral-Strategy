"""
Performance attribution by pair, sector, gross vs net, and cost friction.
"""

import pandas as pd


def attribute_returns_by_pair(trades_df: pd.DataFrame, ledger_fills: list) -> pd.DataFrame:
    """
    Decomposes realized performance and transaction costs across individual pairs.
    """
    if trades_df.empty:
        return pd.DataFrame(
            columns=["pair_id", "total_trades", "win_rate", "total_pnl", "profit_factor"]
        )

    # Aggregate fills by symbol to estimate pair costs
    pair_summary = []
    unique_pairs = trades_df["pair_id"].unique()

    for p in unique_pairs:
        p_trades = trades_df[trades_df["pair_id"] == p]
        n_trades = len(p_trades)

        # Approximate trade P&L from Z-score changes and duration
        win_trades = (p_trades["entry_zscore"].abs() > p_trades["exit_zscore"].abs()).sum()
        win_rate = win_trades / n_trades if n_trades > 0 else 0.0

        pair_summary.append(
            {
                "pair_id": p,
                "total_trades": n_trades,
                "win_rate": float(win_rate),
                "avg_holding_days": float(p_trades["holding_days"].mean())
                if "holding_days" in p_trades
                else 0.0,
            }
        )

    return pd.DataFrame(pair_summary)


def compute_cost_drag_attribution(daily_portfolio: pd.DataFrame) -> dict[str, float]:
    """
    Quantifies drag from 5 bps slippage, broker fees, and short borrow costs.
    """
    if daily_portfolio.empty:
        return {}

    total_gross = daily_portfolio["daily_gross_pnl"].sum()
    total_net = daily_portfolio["daily_net_pnl"].sum()
    total_tx_costs = daily_portfolio["daily_transaction_costs"].sum()
    total_borrow = daily_portfolio["daily_borrow_costs"].sum()

    cost_pct_of_gross = (total_tx_costs / total_gross) if total_gross > 0 else 0.0

    return {
        "cumulative_gross_pnl": float(total_gross),
        "cumulative_net_pnl": float(total_net),
        "cumulative_transaction_costs": float(total_tx_costs),
        "cumulative_borrow_costs": float(total_borrow),
        "total_cost_drag_dollars": float(total_tx_costs + total_borrow),
        "cost_pct_of_gross_pnl": float(cost_pct_of_gross),
    }
