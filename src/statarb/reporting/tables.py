"""
Tabular summary formatting for console and markdown research reports.
"""

from typing import Any

import pandas as pd
from tabulate import tabulate


def format_performance_summary_table(metrics: dict[str, Any]) -> str:
    """Formats headline performance table in GitHub Markdown syntax."""
    rows = [
        ["Total Out-of-Sample Return", f"{metrics.get('cumulative_return', 0.0):.2%}"],
        ["CAGR (Annualized Return)", f"{metrics.get('cagr', 0.0):.2%}"],
        ["Annualized Volatility", f"{metrics.get('annualized_volatility', 0.0):.2%}"],
        ["Annualized Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0.0):.2f}"],
        ["Sortino Ratio", f"{metrics.get('sortino_ratio', 0.0):.2f}"],
        ["Calmar Ratio", f"{metrics.get('calmar_ratio', 0.0):.2f}"],
        ["Maximum Drawdown", f"{metrics.get('max_drawdown', 0.0):.2%}"],
        ["Max Drawdown Duration", f"{metrics.get('max_drawdown_days', 0)} sessions"],
        ["Value at Risk (VaR 95%)", f"{metrics.get('var_95', 0.0):.2%}"],
        ["Conditional VaR (CVaR 95%)", f"{metrics.get('cvar_95', 0.0):.2%}"],
        ["Annualized Turnover (Two-Way)", f"{metrics.get('annualized_turnover', 0.0):.1f}x"],
        ["Total Executed Trades", f"{metrics.get('total_trades', 0)}"],
        ["Trade Win Rate", f"{metrics.get('win_rate', 0.0):.1%}"],
        ["Total Transaction Costs", f"${metrics.get('total_transaction_costs', 0.0):,.2f}"],
        ["Total Short Borrow Fees", f"${metrics.get('total_borrow_costs', 0.0):,.2f}"],
    ]
    return tabulate(rows, headers=["Metric", "Measured Value"], tablefmt="github")


def format_monthly_table(monthly_df: pd.DataFrame) -> str:
    """Formats monthly returns table in percentage strings."""
    if monthly_df.empty:
        return "No monthly returns data available."
    formatted = monthly_df.map(lambda x: f"{x:.1%}" if pd.notnull(x) else "-")
    return tabulate(formatted, headers="keys", tablefmt="github")
