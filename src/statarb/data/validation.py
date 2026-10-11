"""
Data quality assurance, validation rules, and auditing report.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import pandas as pd

from statarb.exceptions import DataValidationError


@dataclass
class SymbolQualityMetrics:
    symbol: str
    total_observations: int
    missing_count: int
    missing_pct: float
    duplicate_count: int
    non_positive_price_count: int
    negative_volume_count: int
    stale_price_count: int
    is_valid: bool
    exclusion_reason: str | None = None


@dataclass
class DataQualityReport:
    generated_at_utc: str
    total_symbols_evaluated: int
    valid_symbols_count: int
    excluded_symbols_count: int
    date_min: str | None
    date_max: str | None
    symbols: dict[str, SymbolQualityMetrics] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class DataValidator:
    """
    Validates market data schemas, physical value domains, and temporal consistency.
    """

    REQUIRED_COLUMNS = ["date", "symbol", "close", "volume"]

    def __init__(
        self,
        max_missing_pct: float = 0.02,
        max_stale_days: int = 10,
        max_daily_return_abs: float = 2.0,  # 200% return flag
    ):
        self.max_missing_pct = max_missing_pct
        self.max_stale_days = max_stale_days
        self.max_daily_return_abs = max_daily_return_abs

    def validate_dataframe(self, df: pd.DataFrame) -> DataQualityReport:
        """
        Runs comprehensive data quality validation across all symbols in the dataset.
        """
        # 1. Schema check
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                raise DataValidationError(f"Missing mandatory column: '{col}'")

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        all_symbols = df["symbol"].unique()
        report_metrics: dict[str, SymbolQualityMetrics] = {}
        valid_symbols = []
        excluded_symbols = []

        date_min = df["date"].min().strftime("%Y-%m-%d") if not df.empty else None
        date_max = df["date"].max().strftime("%Y-%m-%d") if not df.empty else None

        for sym in all_symbols:
            sym_df = df[df["symbol"] == sym].sort_values("date")
            total_obs = len(sym_df)

            # Check duplicates
            dup_count = sym_df.duplicated(subset=["date"]).sum()

            # Check price column
            price_col = "adj_close" if "adj_close" in sym_df.columns else "close"
            non_positive_prices = (sym_df[price_col] <= 0).sum()
            missing_prices = sym_df[price_col].isna().sum()

            # Check volume
            neg_volume = (sym_df["volume"] < 0).sum()

            # Check stale prices (consecutive identical prices)
            price_diffs = sym_df[price_col].diff().abs()
            stale_streaks = (price_diffs == 0).astype(int)
            # Find max consecutive zeros
            stale_max = 0
            curr_streak = 0
            for val in stale_streaks:
                if val == 1:
                    curr_streak += 1
                    stale_max = max(stale_max, curr_streak)
                else:
                    curr_streak = 0

            # Missing percentage
            missing_pct = (missing_prices / total_obs) if total_obs > 0 else 1.0

            exclusion_reason = None
            if non_positive_prices > 0:
                exclusion_reason = f"Contains {non_positive_prices} non-positive prices"
            elif neg_volume > 0:
                exclusion_reason = f"Contains {neg_volume} negative volume records"
            elif missing_pct > self.max_missing_pct:
                exclusion_reason = (
                    f"Missing data {missing_pct:.1%} exceeds tolerance {self.max_missing_pct:.1%}"
                )
            elif stale_max >= self.max_stale_days:
                exclusion_reason = (
                    f"Stale prices detected with {stale_max} consecutive unchanged sessions"
                )

            is_valid = exclusion_reason is None

            if is_valid:
                valid_symbols.append(sym)
            else:
                excluded_symbols.append(sym)

            report_metrics[sym] = SymbolQualityMetrics(
                symbol=sym,
                total_observations=total_obs,
                missing_count=int(missing_prices),
                missing_pct=float(missing_pct),
                duplicate_count=int(dup_count),
                non_positive_price_count=int(non_positive_prices),
                negative_volume_count=int(neg_volume),
                stale_price_count=int(stale_max),
                is_valid=is_valid,
                exclusion_reason=exclusion_reason,
            )

        return DataQualityReport(
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            total_symbols_evaluated=len(all_symbols),
            valid_symbols_count=len(valid_symbols),
            excluded_symbols_count=len(excluded_symbols),
            date_min=date_min,
            date_max=date_max,
            symbols=report_metrics,
        )
