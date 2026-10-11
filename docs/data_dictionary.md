# Data Dictionary & Schema Specifications

## 1. Overview
This document specifies the exact schemas, column naming conventions, data types, physical units, and constraints for all data structures utilized across the Statistical Arbitrage & Equity Market Neutral research platform.

---

## 2. Ingestion & Market Data Schemas

### 2.1 Daily OHLCV Price Records (`data/processed/prices.parquet`)
| Column Name | Type | Unit / Format | Description | Constraints |
|:---|:---|:---|:---|:---|
| `date` | `datetime64[ns]` | ISO 8601 (YYYY-MM-DD) | Trading date | Monotonic, business days |
| `symbol` | `string` | Ticker symbol | Equity ticker (e.g., AAPL, MSFT) | Non-empty |
| `open` | `float64` | USD ($) | Unadjusted session open price | $> 0$ |
| `high` | `float64` | USD ($) | Unadjusted session high price | $\ge \text{low}, \ge \text{open}, \ge \text{close}$ |
| `low` | `float64` | USD ($) | Unadjusted session low price | $> 0, \le \text{high}$ |
| `close` | `float64` | USD ($) | Unadjusted session close price | $> 0$ |
| `adj_close` | `float64` | USD ($) | Split- and dividend-adjusted close | $> 0$ |
| `volume` | `int64` | Share count | Executed share trading volume | $\ge 0$ |
| `dollar_volume` | `float64` | USD ($) | $\text{close} \times \text{volume}$ | $\ge 0$ |

---

## 3. Pair Selection & Econometric Diagnostics (`reports/results/.../selected_pairs.csv`)

| Column Name | Type | Description |
|:---|:---|:---|
| `pair_id` | `string` | Deterministic identifier (e.g., `TECH_AAPL_MSFT_v1`) |
| `asset_a` | `string` | Dependent asset symbol $y_t$ |
| `asset_b` | `string` | Independent asset symbol $x_t$ |
| `sector` | `string` | GICS sector classification |
| `sample_size` | `int64` | Number of observations in formation window |
| `alpha` | `float64` | Estimated intercept from $\log(P_A) = \alpha + \beta \log(P_B) + \epsilon$ |
| `beta` | `float64` | Estimated long-run cointegrating hedge ratio |
| `adf_stat` | `float64` | Augmented Dickey-Fuller test statistic on residuals |
| `adf_pvalue` | `float64` | Unadjusted asymptotic p-value |
| `fdr_qvalue` | `float64` | Benjamini-Hochberg FDR-adjusted q-value |
| `johansen_trace` | `float64` | Johansen trace test statistic for $r=0$ |
| `johansen_crit95` | `float64` | Johansen 95% critical value |
| `hurst_exponent` | `float64` | Rescaled Range / variance-time Hurst exponent estimate ($< 0.5$ anti-persistent) |
| `half_life` | `float64` | Ornstein-Uhlenbeck mean-reversion half-life (trading days) |
| `spread_std` | `float64` | Formation window spread standard deviation $\sigma_S$ |
| `stability_cv` | `float64` | Rolling hedge ratio coefficient of variation |
| `is_selected` | `bool` | Final binary inclusion flag |
| `exclusion_reason` | `string` | Reason code if rejected (`P_VAL_HIGH`, `HURST_HIGH`, `HALF_LIFE_BOUNDS`, `CV_UNSTABLE`) |

---

## 4. Signal State Machine Records (`signals.parquet`)

| Column Name | Type | Description |
|:---|:---|:---|
| `date` | `datetime64[ns]` | Trading date |
| `pair_id` | `string` | Pair identifier |
| `spread` | `float64` | Observed spread $S_t = \log(P_A) - \alpha - \beta \log(P_B)$ |
| `rolling_mean` | `float64` | Causal rolling spread mean $\mu_{t-1}$ (excluding day $t$) |
| `rolling_std` | `float64` | Causal rolling spread standard deviation $\sigma_{t-1}$ |
| `z_score` | `float64` | Lagged-normalized Z-score: $(S_t - \mu_{t-1}) / \sigma_{t-1}$ |
| `state` | `string` | Position state (`FLAT`, `LONG_SPREAD`, `SHORT_SPREAD`, `CLOSING`, `DISABLED`) |
| `target_weight_a` | `float64` | Target portfolio weight for Asset A |
| `target_weight_b` | `float64` | Target portfolio weight for Asset B |

---

## 5. Trade Ledger Records (`trades.csv`)

| Column Name | Type | Description |
|:---|:---|:---|
| `trade_id` | `string` | Unique UUID |
| `pair_id` | `string` | Pair identifier |
| `entry_date` | `datetime64[ns]` | Date position was opened |
| `exit_date` | `datetime64[ns]` | Date position was closed |
| `direction` | `string` | Position direction (`LONG_SPREAD`, `SHORT_SPREAD`) |
| `holding_days` | `int64` | Elapsed trading sessions |
| `entry_zscore` | `float64` | Z-score at entry signal |
| `exit_zscore` | `float64` | Z-score at exit signal |
| `gross_pnl` | `float64` | Mark-to-market Gross P&L ($) before costs |
| `slippage_cost` | `float64` | Total 5 bps slippage paid ($) on entry and exit |
| `commission_cost`| `float64` | Total broker commissions paid ($) |
| `borrow_cost` | `float64` | Short borrow financing costs ($) |
| `net_pnl` | `float64` | Realized Net P&L: $\text{Gross PnL} - \text{Slippage} - \text{Commissions} - \text{Borrow}$ |
| `return_pct` | `float64` | Net P&L divided by allocated capital |
| `exit_reason` | `string` | Trigger code (`MEAN_REVERSION`, `STOP_LOSS`, `MAX_HOLDING`, `MODEL_BREAK`) |

---

## 6. Daily Portfolio Output (`daily_portfolio.parquet` / `equity_curve.csv`)

| Column Name | Type | Description |
|:---|:---|:---|
| `date` | `datetime64[ns]` | Trading session date |
| `cash` | `float64` | Unencumbered cash balance ($) |
| `holdings_val` | `float64` | Net market value of long + short positions ($) |
| `portfolio_equity`| `float64` | Total Net Asset Value (NAV): $\text{Cash} + \text{Holdings}$ |
| `gross_exposure` | `float64` | $\sum |\text{Market Value}| / \text{Equity}$ |
| `net_exposure` | `float64` | $\sum \text{Market Value} / \text{Equity}$ |
| `daily_gross_pnl` | `float64` | Daily change in valuation before trading frictions ($) |
| `daily_costs` | `float64` | Daily slippage, commissions, and borrow fees ($) |
| `daily_net_return`| `float64` | $(E_t - E_{t-1}) / E_{t-1}$ after all costs |
| `daily_turnover` | `float64` | Daily Traded Notional / Portfolio Equity |
| `active_pairs` | `int64` | Number of simultaneous active pairs |
| `drawdown` | `float64` | $(E_t - \max_{s \le t} E_s) / \max_{s \le t} E_s$ |
