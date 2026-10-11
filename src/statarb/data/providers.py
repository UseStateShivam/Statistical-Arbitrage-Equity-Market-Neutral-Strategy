"""
Data provider abstractions and implementations (Synthetic, Parquet, Yahoo).
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from statarb.data.normalization import normalize_ohlcv_dataframe
from statarb.exceptions import DataValidationError


class BaseDataProvider(ABC):
    """Abstract interface defining market data retrieval contract."""

    @abstractmethod
    def fetch_ohlcv(self, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Retrieves daily OHLCV dataframe in normalized long format."""


class LocalParquetProvider(BaseDataProvider):
    """Reads processed Parquet datasets from local disk storage."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def fetch_ohlcv(self, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        target_file = self.data_dir / "prices.parquet"
        if not target_file.exists():
            raise FileNotFoundError(f"Parquet file not found at {target_file}")

        df = pd.read_parquet(target_file)
        df = normalize_ohlcv_dataframe(df)

        if symbols:
            df = df[df["symbol"].isin([s.upper() for s in symbols])]

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
        return df.reset_index(drop=True)


class SyntheticDataProvider(BaseDataProvider):
    """
    Generates deterministic, institutional-grade synthetic equity data
    with mathematically rigorous cointegrating relationships, sector groupings,
    and realistic volume/liquidity dynamics.
    """

    DEFAULT_SECTORS = {
        "TECH": ["AAPL", "MSFT", "GOOGL", "NVDA", "AMD"],
        "FIN": ["JPM", "BAC", "WFC", "C", "GS"],
        "ENERGY": ["XOM", "CVX", "COP", "SLB", "OXY"],
        "HEALTH": ["JNJ", "PFE", "MRK", "ABT", "BMY"],
    }

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fetch_ohlcv(
        self, symbols: list[str], start_date: str = "2020-01-01", end_date: str = "2023-12-31"
    ) -> pd.DataFrame:
        rng = np.random.RandomState(self.seed)
        date_range = pd.bdate_range(start=start_date, end=end_date)
        n_days = len(date_range)

        # Market benchmark factor (SPY)
        mkt_returns = rng.normal(0.0003, 0.012, n_days)
        spy_prices = 300.0 * np.exp(np.cumsum(mkt_returns))

        data_rows = []

        # Add SPY benchmark rows
        for t, dt in enumerate(date_range):
            close = spy_prices[t]
            vol = rng.lognormal(18.0, 0.3)
            data_rows.append(
                {
                    "date": dt,
                    "symbol": "SPY",
                    "open": close * (1 + rng.normal(0, 0.003)),
                    "high": close * (1 + abs(rng.normal(0, 0.008))),
                    "low": close * (1 - abs(rng.normal(0, 0.008))),
                    "close": close,
                    "adj_close": close,
                    "volume": int(vol),
                    "dollar_volume": close * vol,
                }
            )

        # Generate Sector Equities
        symbols if symbols else [s for sec in self.DEFAULT_SECTORS.values() for s in sec]

        for sec_symbols in self.DEFAULT_SECTORS.values():
            # Sector factor
            sec_factor = rng.normal(0.0002, 0.015, n_days)
            sec_base_log_price = np.cumsum(0.7 * mkt_returns + 0.3 * sec_factor)

            # Define specific cointegrated pairs within the sector:
            # Pair 1: sym0 & sym1 cointegrated via OU process
            # Pair 2: sym2 & sym3 cointegrated via OU process
            # sym4: independent random walk
            for i, sym in enumerate(sec_symbols):
                if symbols and sym not in symbols:
                    continue

                if i == 0:
                    # Anchor asset: geometric random walk driven by sector + idio
                    idio = rng.normal(0, 0.014, n_days)
                    log_p = 4.0 + sec_base_log_price + np.cumsum(idio)
                elif i == 1:
                    # Cointegrated partner to sym0:
                    # log(P1) = alpha + beta * log(P0) + spread
                    # spread is an Ornstein-Uhlenbeck process with theta = 0.05 (half life ~ 14 days)
                    theta = 0.05
                    sigma_ou = 0.015
                    spread = np.zeros(n_days)
                    for t in range(1, n_days):
                        spread[t] = spread[t - 1] - theta * spread[t - 1] + rng.normal(0, sigma_ou)

                    alpha = 0.5
                    beta = 1.15
                    log_p0 = np.array(
                        [r["close"] for r in data_rows if r["symbol"] == sec_symbols[0]]
                    )
                    log_p0 = np.log(log_p0)
                    log_p = alpha + beta * log_p0 + spread
                elif i == 2:
                    # Second anchor asset
                    idio = rng.normal(0, 0.015, n_days)
                    log_p = 3.5 + sec_base_log_price + np.cumsum(idio)
                elif i == 3:
                    # Cointegrated partner to sym2:
                    theta = 0.06
                    sigma_ou = 0.016
                    spread = np.zeros(n_days)
                    for t in range(1, n_days):
                        spread[t] = spread[t - 1] - theta * spread[t - 1] + rng.normal(0, sigma_ou)
                    alpha = -0.2
                    beta = 0.90
                    log_p2 = np.array(
                        [r["close"] for r in data_rows if r["symbol"] == sec_symbols[2]]
                    )
                    log_p2 = np.log(log_p2)
                    log_p = alpha + beta * log_p2 + spread
                else:
                    # Independent random walk asset in sector (high correlation, NO cointegration)
                    idio = rng.normal(0, 0.018, n_days)
                    log_p = 3.8 + sec_base_log_price + np.cumsum(idio)

                prices = np.exp(log_p)

                for t, dt in enumerate(date_range):
                    close = float(prices[t])
                    # High/low consistent with open/close
                    open_p = close * (1 + rng.normal(0, 0.004))
                    high_p = max(open_p, close) * (1 + abs(rng.normal(0, 0.007)))
                    low_p = min(open_p, close) * (1 - abs(rng.normal(0, 0.007)))
                    vol = int(rng.lognormal(14.5, 0.4))
                    data_rows.append(
                        {
                            "date": dt,
                            "symbol": sym,
                            "open": open_p,
                            "high": high_p,
                            "low": low_p,
                            "close": close,
                            "adj_close": close,
                            "volume": vol,
                            "dollar_volume": close * vol,
                        }
                    )

        df = pd.DataFrame(data_rows)
        return normalize_ohlcv_dataframe(df)


class YahooFinanceProvider(BaseDataProvider):
    """Downloads daily market data via yfinance with caching."""

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def fetch_ohlcv(self, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError("yfinance must be installed to use YahooFinanceProvider.") from e

        all_dfs = []
        for sym in symbols:
            ticker = yf.Ticker(sym)
            hist = ticker.history(start=start_date, end=end_date, auto_adjust=False)
            if hist.empty:
                continue
            hist = hist.reset_index()
            hist.columns = [c.lower() for c in hist.columns]
            hist["symbol"] = sym.upper()
            all_dfs.append(hist)

        if not all_dfs:
            raise DataValidationError("No data retrieved from Yahoo Finance for requested symbols.")

        df = pd.concat(all_dfs, ignore_index=True)
        return normalize_ohlcv_dataframe(df)
