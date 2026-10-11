"""
Unit tests for configuration validation and schema enforcement.
"""

import pytest

from statarb.config import AppConfig, HalfLifeConfig, SignalsConfig
from statarb.exceptions import ConfigurationError


def test_valid_config_loading(app_config: AppConfig):
    assert app_config.experiment.id == "exp_baseline_v1"
    assert app_config.execution.slippage_bps == 5.0
    assert app_config.signals.z_entry > app_config.signals.z_exit
    assert app_config.signals.z_stop > app_config.signals.z_entry


def test_invalid_signal_thresholds():
    # z_entry <= z_exit
    with pytest.raises(Exception):
        SignalsConfig(z_entry=1.0, z_exit=1.5)

    # z_stop <= z_entry
    with pytest.raises(Exception):
        SignalsConfig(z_entry=2.0, z_exit=0.0, z_stop=1.8)


def test_invalid_half_life_bounds():
    with pytest.raises(Exception):
        HalfLifeConfig(min_half_life=30.0, max_half_life=10.0)


def test_nonexistent_config_file():
    with pytest.raises(ConfigurationError):
        AppConfig.from_yaml("nonexistent_path_xyz.yaml")
