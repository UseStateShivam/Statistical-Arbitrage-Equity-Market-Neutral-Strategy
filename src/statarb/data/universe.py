"""
Point-in-time eligible universe construction and liquidity filtering.
"""

from dataclasses import dataclass

import pandas as pd

from statarb.config import UniverseConfig


@dataclass
class UniverseAuditRecord:
    symbol: str
    sector: str
    median_price: float
    median_dollar_volume: float
    missing_pct: float
    total_days: int
    is_eligible: bool
    rejection_reason: str | None = None


class UniverseFilter:
    """
    Constructs an eligible universe strictly point-in-time without look-ahead bias.
    """

    DEFAULT_SECTOR_MAP = {
        "AAPL": "Information Technology",
        "MSFT": "Information Technology",
        "GOOGL": "Information Technology",
        "NVDA": "Information Technology",
        "AMD": "Information Technology",
        "INTC": "Information Technology",
        "JPM": "Financials",
        "BAC": "Financials",
        "WFC": "Financials",
        "C": "Financials",
        "GS": "Financials",
        "MS": "Financials",
        "XOM": "Energy",
        "CVX": "Energy",
        "COP": "Energy",
        "SLB": "Energy",
        "OXY": "Energy",
        "EOG": "Energy",
        "JNJ": "Healthcare",
        "PFE": "Healthcare",
        "MRK": "Healthcare",
        "ABT": "Healthcare",
        "BMY": "Healthcare",
        "AMGN": "Healthcare",
        "SPY": "Benchmark",
    }

    def __init__(self, config: UniverseConfig, sector_map: dict[str, str] | None = None):
        self.config = config
        self.sector_map = sector_map or self.DEFAULT_SECTOR_MAP

    def filter_universe(
        self, df: pd.DataFrame, as_of_date: pd.Timestamp, lookback_days: int = 252
    ) -> list[UniverseAuditRecord]:
        """
        Filters eligible equities using exclusively data observed on or prior to as_of_date.
        """
        df = df[df["date"] <= as_of_date].copy()
        unique_dates = sorted(df["date"].unique())
        if len(unique_dates) > lookback_days:
            cutoff_date = unique_dates[-lookback_days]
            df = df[df["date"] >= cutoff_date]

        symbols = df["symbol"].unique()
        audit_records: list[UniverseAuditRecord] = []

        for sym in symbols:
            # Skip benchmark symbol from pair trading universe
            if sym.upper() == "SPY":
                continue

            sym_df = df[df["symbol"] == sym]
            total_days = len(sym_df)
            expected_days = (
                len(unique_dates[-lookback_days:])
                if len(unique_dates) > lookback_days
                else len(unique_dates)
            )

            missing_count = expected_days - total_days
            missing_pct = max(0.0, missing_count / expected_days) if expected_days > 0 else 1.0

            price_col = "adj_close" if "adj_close" in sym_df.columns else "close"
            median_price = float(sym_df[price_col].median()) if not sym_df.empty else 0.0

            # Compute median dollar volume over last 60 sessions
            recent_df = sym_df.tail(60)
            if "dollar_volume" in recent_df.columns:
                median_dollar_vol = float(recent_df["dollar_volume"].median())
            elif "volume" in recent_df.columns and price_col in recent_df.columns:
                median_dollar_vol = float((recent_df[price_col] * recent_df["volume"]).median())
            else:
                median_dollar_vol = 0.0

            sector = self.sector_map.get(sym, "Unknown")

            # Check criteria
            rejection_reason = None
            if total_days < self.config.min_history_days:
                rejection_reason = (
                    f"History length ({total_days}) < min ({self.config.min_history_days})"
                )
            elif median_price < self.config.min_price:
                rejection_reason = (
                    f"Median price (${median_price:.2f}) < min (${self.config.min_price:.2f})"
                )
            elif median_dollar_vol < self.config.min_median_dollar_volume:
                rejection_reason = f"Dollar volume (${median_dollar_vol:,.0f}) < min (${self.config.min_median_dollar_volume:,.0f})"
            elif missing_pct > self.config.max_missing_pct:
                rejection_reason = (
                    f"Missingness ({missing_pct:.1%}) > max ({self.config.max_missing_pct:.1%})"
                )
            elif self.config.symbols and sym not in self.config.symbols:
                rejection_reason = "Symbol not in configured whitelist"

            is_eligible = rejection_reason is None

            audit_records.append(
                UniverseAuditRecord(
                    symbol=sym,
                    sector=sector,
                    median_price=median_price,
                    median_dollar_volume=median_dollar_vol,
                    missing_pct=missing_pct,
                    total_days=total_days,
                    is_eligible=is_eligible,
                    rejection_reason=rejection_reason,
                )
            )

        return audit_records

    def get_eligible_symbols(self, audit_records: list[UniverseAuditRecord]) -> list[str]:
        """Extracts list of eligible ticker symbols."""
        return [r.symbol for r in audit_records if r.is_eligible]
