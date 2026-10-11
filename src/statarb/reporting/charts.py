"""
Publication-quality financial visualizations with matplotlib and seaborn.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set clean aesthetic style
plt.style.use(
    "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10


def plot_equity_curve(daily_df: pd.DataFrame, output_path: Path | str, dpi: int = 300) -> None:
    """Plots Out-of-Sample Net vs Gross Equity Curve and Transaction Cost Drag."""
    _fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    equity = daily_df["portfolio_equity"]
    initial_cap = equity.iloc[0]
    cumulative_costs = daily_df["daily_transaction_costs"].cumsum()
    gross_equity = equity + cumulative_costs

    ax1.plot(
        daily_df.index,
        equity,
        label="Net Portfolio Equity (After 5 bps Slippage & Fees)",
        color="#1f77b4",
        lw=1.8,
    )
    ax1.plot(
        daily_df.index,
        gross_equity,
        label="Gross Equity (Zero Frictions)",
        color="#2ca02c",
        lw=1.2,
        ls="--",
    )
    ax1.axhline(initial_cap, color="gray", lw=0.8, ls=":")
    ax1.set_title("Out-of-Sample Portfolio Equity Curve: Gross vs Net", fontweight="bold")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left")

    # Lower panel: cumulative costs
    ax2.plot(
        daily_df.index,
        cumulative_costs,
        label="Cumulative Slippage & Execution Costs ($)",
        color="#d62728",
        lw=1.5,
    )
    ax2.set_title("Cumulative Friction Impact", fontweight="bold")
    ax2.set_ylabel("Costs ($)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_drawdowns(daily_df: pd.DataFrame, output_path: Path | str, dpi: int = 300) -> None:
    """Plots underwater drawdown percentage over time."""
    _fig, ax = plt.subplots(figsize=(11, 4.5))

    equity = daily_df["portfolio_equity"]
    peak = equity.cummax()
    dd_pct = (equity - peak) / peak * 100.0

    ax.fill_between(daily_df.index, dd_pct, 0, color="#d62728", alpha=0.3, label="Drawdown (%)")
    ax.plot(daily_df.index, dd_pct, color="#d62728", lw=1.2)
    ax.set_title("Underwater Portfolio Drawdown Profile", fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_rolling_sharpe(
    daily_df: pd.DataFrame, output_path: Path | str, window: int = 63, dpi: int = 300
) -> None:
    """Plots 63-day rolling annualized Sharpe ratio."""
    _fig, ax = plt.subplots(figsize=(11, 4.5))

    returns = daily_df["daily_net_return"]
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std(ddof=1)
    rolling_sharpe = np.sqrt(252) * (rolling_mean / rolling_std)

    ax.plot(
        daily_df.index,
        rolling_sharpe,
        color="#9467bd",
        lw=1.5,
        label=f"{window}-Day Rolling Sharpe",
    )
    ax.axhline(0.0, color="gray", lw=0.8, ls=":")
    ax.axhline(1.0, color="green", lw=0.8, ls="--", alpha=0.6, label="Sharpe = 1.0 Benchmark")
    ax.set_title("Rolling Risk-Adjusted Performance (Annualized Sharpe)", fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_monthly_heatmap(monthly_df: pd.DataFrame, output_path: Path | str, dpi: int = 300) -> None:
    """Plots monthly returns percentage heatmap."""
    if monthly_df.empty:
        return

    _fig, ax = plt.subplots(figsize=(10, max(3, len(monthly_df) * 0.8)))
    heatmap_data = monthly_df * 100.0  # in percent

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=0.0,
        cbar_kws={"label": "Monthly Return (%)"},
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Monthly Return Heatmap (%)", fontweight="bold")
    ax.set_ylabel("Year")
    ax.set_xlabel("Month")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_exposures(exposures_df: pd.DataFrame, output_path: Path | str, dpi: int = 300) -> None:
    """Plots gross and net portfolio exposure over time."""
    if exposures_df.empty:
        return

    _fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(
        exposures_df.index,
        exposures_df["gross_exposure"] * 100.0,
        label="Gross Exposure (%)",
        color="#ff7f0e",
        lw=1.5,
    )
    ax.plot(
        exposures_df.index,
        exposures_df["net_exposure"] * 100.0,
        label="Net Exposure (%)",
        color="#1f77b4",
        lw=1.5,
    )
    ax.axhline(0.0, color="gray", lw=0.8, ls=":")
    ax.axhline(100.0, color="gray", lw=0.8, ls="--", label="100% Gross Target")
    ax.set_title("Portfolio Leverage and Market Neutrality Profile", fontweight="bold")
    ax.set_ylabel("Exposure (% of Equity)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_spread_zscore(
    pair_id: str, signals_df: pd.DataFrame, output_path: Path | str, dpi: int = 300
) -> None:
    """Plots spread, Bollinger bands, Z-score, and executed trades."""
    if signals_df.empty or "z_score" not in signals_df.columns:
        return

    _fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 2]}
    )

    dates = signals_df.index
    z = signals_df["z_score"]

    # Upper panel: Z-score with entry/exit/stop bands
    ax1.plot(dates, z, label="Rolling Z-Score", color="#1f77b4", lw=1.5)
    ax1.axhline(+2.0, color="#d62728", ls="--", lw=1.0, label="Short Entry (+2.0)")
    ax1.axhline(-2.0, color="#2ca02c", ls="--", lw=1.0, label="Long Entry (-2.0)")
    ax1.axhline(0.0, color="gray", ls=":", lw=0.8, label="Exit Target (0.0)")
    ax1.axhline(+3.5, color="black", ls=":", lw=1.0, label="Stop Loss (+/- 3.5)")
    ax1.axhline(-3.5, color="black", ls=":", lw=1.0)
    ax1.set_title(f"Pair Spread Diagnostics & State Transitions: {pair_id}", fontweight="bold")
    ax1.set_ylabel("Z-Score")
    ax1.legend(loc="upper left", ncol=2)

    # Lower panel: position direction (+1, -1, 0)
    direction = signals_df["direction"]
    ax2.step(dates, direction, label="Position State", color="#ff7f0e", lw=1.5, where="post")
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(["Short Spread", "Flat", "Long Spread"])
    ax2.set_ylabel("State")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_cost_sensitivity(
    sensitivity_df: pd.DataFrame, output_path: Path | str, dpi: int = 300
) -> None:
    """Plots Sharpe ratio and annualized return across slippage scenarios (0 to 20 bps)."""
    if sensitivity_df.empty:
        return

    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    slippage = sensitivity_df["slippage_bps"]
    sharpe = sensitivity_df["sharpe_ratio"]
    cagr = sensitivity_df["cagr"] * 100.0

    ax1.plot(slippage, sharpe, marker="o", color="#1f77b4", lw=1.8)
    ax1.axvline(5.0, color="red", ls="--", lw=1.0, label="Baseline (5 bps)")
    ax1.axhline(0.0, color="gray", ls=":", lw=0.8)
    ax1.set_title("Sharpe Ratio vs Execution Slippage", fontweight="bold")
    ax1.set_xlabel("Slippage (basis points per side)")
    ax1.set_ylabel("Sharpe Ratio")
    ax1.legend()

    ax2.plot(slippage, cagr, marker="s", color="#2ca02c", lw=1.8)
    ax2.axvline(5.0, color="red", ls="--", lw=1.0, label="Baseline (5 bps)")
    ax2.axhline(0.0, color="gray", ls=":", lw=0.8)
    ax2.set_title("CAGR Return vs Execution Slippage", fontweight="bold")
    ax2.set_xlabel("Slippage (basis points per side)")
    ax2.set_ylabel("Annualized Net Return (%)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
