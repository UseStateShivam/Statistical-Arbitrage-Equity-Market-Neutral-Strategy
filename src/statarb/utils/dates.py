"""
Date management, calendar alignment, and walk-forward scheduling utilities.
"""

from typing import NamedTuple

import pandas as pd


class WalkForwardWindow(NamedTuple):
    """Encapsulates time bounds for a single chronological walk-forward fold."""

    window_id: int
    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    trading_start: pd.Timestamp
    trading_end: pd.Timestamp


def generate_walk_forward_schedule(
    trading_dates: pd.DatetimeIndex,
    formation_days: int = 252,
    trading_days: int = 63,
    step_days: int = 63,
) -> list[WalkForwardWindow]:
    """
    Constructs strictly chronological, non-overlapping walk-forward out-of-sample trading windows.
    Zero future information leaks into the formation periods.
    """
    unique_dates = sorted(trading_dates.unique())
    total_dates = len(unique_dates)

    schedule: list[WalkForwardWindow] = []
    window_id = 1

    start_idx = 0
    while start_idx + formation_days + trading_days <= total_dates:
        formation_start_idx = start_idx
        formation_end_idx = start_idx + formation_days - 1

        trading_start_idx = formation_end_idx + 1
        trading_end_idx = min(formation_end_idx + trading_days, total_dates - 1)

        schedule.append(
            WalkForwardWindow(
                window_id=window_id,
                formation_start=unique_dates[formation_start_idx],
                formation_end=unique_dates[formation_end_idx],
                trading_start=unique_dates[trading_start_idx],
                trading_end=unique_dates[trading_end_idx],
            )
        )

        window_id += 1
        start_idx += step_days

    return schedule
