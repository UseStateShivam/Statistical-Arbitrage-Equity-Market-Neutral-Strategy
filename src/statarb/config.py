"""
Validated configuration models using Pydantic.
Centralizes all research, risk, execution, and backtesting parameters.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from statarb.exceptions import ConfigurationError


class ExperimentConfig(BaseModel):
    id: str = "exp_baseline_v1"
    description: str = "Baseline Walk-Forward Statistical Arbitrage"
    random_seed: int = 42
    log_level: str = "INFO"


class DataConfig(BaseModel):
    provider: Literal["synthetic", "parquet", "yahoo"] = "synthetic"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    sample_dir: str = "data/sample"
    start_date: str = "2020-01-01"
    end_date: str = "2023-12-31"
    benchmark_symbol: str = "SPY"
    cache_enabled: bool = True


class UniverseConfig(BaseModel):
    min_price: float = Field(default=5.0, gt=0.0)
    min_median_dollar_volume: float = Field(default=5000000.0, ge=0.0)
    max_missing_pct: float = Field(default=0.02, ge=0.0, le=1.0)
    min_history_days: int = Field(default=252, ge=20)
    sector_constraint: bool = True
    symbols: list[str] = Field(default_factory=list)


class CandidateGenerationConfig(BaseModel):
    min_correlation: float = Field(default=0.50, ge=-1.0, le=1.0)
    max_candidates_per_sector: int = Field(default=50, gt=0)
    same_sector_only: bool = True


class CointegrationConfig(BaseModel):
    significance_level: float = Field(default=0.05, gt=0.0, lt=1.0)
    adf_regression: Literal["c", "ct", "n"] = "c"
    adf_lag_selection: Literal["aic", "bic", "t-stat"] = "aic"
    adf_max_lags: int = Field(default=10, ge=0)
    use_engle_granger: bool = True
    use_johansen: bool = True
    johansen_det_order: int = 0
    johansen_k_ar_diff: int = 1
    fdr_correction: bool = True
    fdr_alpha: float = Field(default=0.10, gt=0.0, lt=1.0)


class HurstConfig(BaseModel):
    max_hurst: float = Field(default=0.45, gt=0.0, lt=1.0)
    min_window: int = Field(default=126, ge=20)
    method: Literal["rs", "variance_time"] = "rs"


class HalfLifeConfig(BaseModel):
    min_half_life: float = Field(default=2.0, gt=0.0)
    max_half_life: float = Field(default=63.0, gt=0.0)

    @model_validator(mode="after")
    def validate_half_life_bounds(self) -> "HalfLifeConfig":
        if self.min_half_life >= self.max_half_life:
            raise ConfigurationError(
                f"min_half_life ({self.min_half_life}) must be strictly less than "
                f"max_half_life ({self.max_half_life})"
            )
        return self


class StabilityConfig(BaseModel):
    max_hedge_ratio_cv: float = Field(default=0.25, gt=0.0)
    check_chow_break: bool = True
    chow_significance: float = Field(default=0.01, gt=0.0, lt=1.0)


class ResearchConfig(BaseModel):
    candidate_generation: CandidateGenerationConfig = Field(
        default_factory=CandidateGenerationConfig
    )
    cointegration: CointegrationConfig = Field(default_factory=CointegrationConfig)
    hurst: HurstConfig = Field(default_factory=HurstConfig)
    half_life: HalfLifeConfig = Field(default_factory=HalfLifeConfig)
    stability: StabilityConfig = Field(default_factory=StabilityConfig)


class SignalsConfig(BaseModel):
    spread_lookback: int = Field(default=63, ge=10)
    z_entry: float = Field(default=2.0, gt=0.0)
    z_exit: float = Field(default=0.0)
    z_stop: float = Field(default=3.5, gt=0.0)
    max_holding_days: int = Field(default=30, gt=0)
    envelope_multiplier: float = Field(default=2.0, gt=0.0)
    envelope_confirmation: bool = True

    @model_validator(mode="after")
    def validate_thresholds(self) -> "SignalsConfig":
        if self.z_entry <= self.z_exit:
            raise ConfigurationError(
                f"Entry Z-score ({self.z_entry}) must exceed exit Z-score ({self.z_exit})"
            )
        if self.z_stop <= self.z_entry:
            raise ConfigurationError(
                f"Stop Z-score ({self.z_stop}) must exceed entry Z-score ({self.z_entry})"
            )
        return self


class PortfolioConfig(BaseModel):
    initial_capital: float = Field(default=1000000.0, gt=0.0)
    max_active_pairs: int = Field(default=10, gt=0)
    sizing_method: Literal["inv_vol", "equal_weight"] = "inv_vol"
    max_capital_per_pair: float = Field(default=0.10, gt=0.0, le=1.0)
    max_single_stock_weight: float = Field(default=0.15, gt=0.0, le=1.0)
    max_sector_weight: float = Field(default=0.30, gt=0.0, le=1.0)
    max_gross_exposure: float = Field(default=1.0, gt=0.0)
    target_net_exposure: float = 0.0
    net_exposure_tolerance: float = Field(default=0.05, ge=0.0)
    beta_neutral: bool = True


class ExecutionConfig(BaseModel):
    fill_convention: Literal["next_open", "next_close"] = "next_open"
    slippage_bps: float = Field(default=5.0, ge=0.0)  # 5 bps required baseline
    commission_per_share: float = Field(default=0.001, ge=0.0)
    borrow_fee_annual_bps: float = Field(default=50.0, ge=0.0)
    cash_interest_annual_bps: float = Field(default=200.0, ge=0.0)
    max_volume_pct: float = Field(default=0.01, gt=0.0, le=1.0)


class BacktestConfig(BaseModel):
    mode: Literal["walk_forward", "single_window"] = "walk_forward"
    formation_window_days: int = Field(default=252, ge=60)
    validation_window_days: int = Field(default=63, ge=10)
    trading_window_days: int = Field(default=63, ge=10)
    rebalance_frequency_days: int = Field(default=63, ge=10)
    annualization_factor: int = 252


class ReportingConfig(BaseModel):
    output_dir: str = "reports/results"
    generate_figures: bool = True
    generate_markdown: bool = True
    dpi: int = 300


class AppConfig(BaseModel):
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    signals: SignalsConfig = Field(default_factory=SignalsConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        path = Path(path)
        if not path.exists():
            raise ConfigurationError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f)
        if not isinstance(raw_dict, dict):
            raise ConfigurationError(f"YAML at {path} did not resolve to a dictionary.")
        try:
            return cls(**raw_dict)
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed for {path}: {e}") from e
