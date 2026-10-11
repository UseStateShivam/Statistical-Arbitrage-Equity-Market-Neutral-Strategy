"""
Signal state machine and position transition generator.
"""

from enum import Enum

import numpy as np
import pandas as pd

from statarb.config import SignalsConfig


class PositionState(str, Enum):
    FLAT = "FLAT"
    LONG_SPREAD = "LONG_SPREAD"  # Long A, Short B
    SHORT_SPREAD = "SHORT_SPREAD"  # Short A, Long B
    CLOSING = "CLOSING"  # Exiting position
    DISABLED = "DISABLED"  # Model invalidation


class ExitReason(str, Enum):
    MEAN_REVERSION = "MEAN_REVERSION"
    STOP_LOSS = "STOP_LOSS"
    MAX_HOLDING = "MAX_HOLDING"
    MODEL_BREAK = "MODEL_BREAK"
    END_OF_WINDOW = "END_OF_WINDOW"


class SignalGenerator:
    """
    Simulates causal finite state machine transitions for a single pair spread series.
    """

    def __init__(self, config: SignalsConfig):
        self.config = config

    def generate_signals(
        self, z_scores: pd.Series, spread: pd.Series | None = None
    ) -> pd.DataFrame:
        """
        Processes daily Z-scores through the deterministic position state machine.
        Returns DataFrame with: state, target_direction, is_entry, is_exit, exit_reason.
        """
        n = len(z_scores)
        dates = z_scores.index

        states = []
        directions = []  # +1 (long spread), -1 (short spread), 0 (flat)
        is_entry = []
        is_exit = []
        exit_reasons = []
        holding_days_list = []

        curr_state = PositionState.FLAT
        curr_direction = 0
        holding_days = 0

        z_vals = z_scores.values

        for t in range(n):
            z_t = z_vals[t]
            entry_flag = False
            exit_flag = False
            exit_reason = None

            if np.isnan(z_t):
                # Cannot trade on NaN
                states.append(curr_state.value)
                directions.append(curr_direction)
                is_entry.append(False)
                is_exit.append(False)
                exit_reasons.append(None)
                holding_days_list.append(holding_days)
                continue

            if curr_state == PositionState.FLAT:
                holding_days = 0
                if z_t >= self.config.z_entry:
                    curr_state = PositionState.SHORT_SPREAD
                    curr_direction = -1
                    entry_flag = True
                elif z_t <= -self.config.z_entry:
                    curr_state = PositionState.LONG_SPREAD
                    curr_direction = 1
                    entry_flag = True

            elif curr_state == PositionState.SHORT_SPREAD:
                holding_days += 1
                # Check exit triggers
                if z_t <= self.config.z_exit:
                    curr_state = PositionState.FLAT
                    curr_direction = 0
                    exit_flag = True
                    exit_reason = ExitReason.MEAN_REVERSION.value
                elif z_t >= self.config.z_stop:
                    curr_state = PositionState.FLAT
                    curr_direction = 0
                    exit_flag = True
                    exit_reason = ExitReason.STOP_LOSS.value
                elif holding_days >= self.config.max_holding_days:
                    curr_state = PositionState.FLAT
                    curr_direction = 0
                    exit_flag = True
                    exit_reason = ExitReason.MAX_HOLDING.value

            elif curr_state == PositionState.LONG_SPREAD:
                holding_days += 1
                # Check exit triggers
                if z_t >= -self.config.z_exit:
                    curr_state = PositionState.FLAT
                    curr_direction = 0
                    exit_flag = True
                    exit_reason = ExitReason.MEAN_REVERSION.value
                elif z_t <= -self.config.z_stop:
                    curr_state = PositionState.FLAT
                    curr_direction = 0
                    exit_flag = True
                    exit_reason = ExitReason.STOP_LOSS.value
                elif holding_days >= self.config.max_holding_days:
                    curr_state = PositionState.FLAT
                    curr_direction = 0
                    exit_flag = True
                    exit_reason = ExitReason.MAX_HOLDING.value

            states.append(curr_state.value)
            directions.append(curr_direction)
            is_entry.append(entry_flag)
            is_exit.append(exit_flag)
            exit_reasons.append(exit_reason)
            holding_days_list.append(holding_days)

        return pd.DataFrame(
            {
                "z_score": z_scores,
                "state": states,
                "direction": directions,
                "is_entry": is_entry,
                "is_exit": is_exit,
                "exit_reason": exit_reasons,
                "holding_days": holding_days_list,
            },
            index=dates,
        )
