# Quantitative Research Report: Statistical Arbitrage & Equity Market Neutral Strategy

**Experiment Identifier:** `exp_baseline_v1`  
**Strategy Class:** Systematic Equity Market Neutral (EMN) Pairs Trading  
**Evaluation Standard:** Chronological Walk-Forward Out-of-Sample Backtesting  
**Execution Friction:** **5 basis points slippage per traded side + commissions + borrow fees**  

---

## 1. Executive Summary

This research report documents the quantitative performance, econometric validity, and risk characteristics of an institutional-grade **Statistical Arbitrage & Equity Market Neutral** strategy. The platform systematically identifies pairs of equities exhibiting statistically robust cointegration under the **Augmented Dickey-Fuller (ADF)** and **Johansen VECM** procedures, corrects for multiple-testing data snooping via **Benjamini-Hochberg False Discovery Rate (FDR)** control, filters candidate spreads via discrete **Ornstein-Uhlenbeck half-life** and **variance-time Hurst exponent** bounds, and executes dollar- and beta-neutral portfolios with realistic market microstructure frictions.

All results presented herein originate strictly from out-of-sample forward trading periods. Zero in-sample parameter tuning or look-ahead leakage is permitted.

---

## 2. Key Out-of-Sample Performance Metrics

| Metric                        | Measured Value   |
|-------------------------------|------------------|
| Total Out-of-Sample Return    | 9.46%            |
| CAGR (Annualized Return)      | 3.06%            |
| Annualized Volatility         | 0.87%            |
| Annualized Sharpe Ratio       | 3.49             |
| Sortino Ratio                 | 2.62             |
| Calmar Ratio                  | 6.00             |
| Maximum Drawdown              | 0.51%            |
| Max Drawdown Duration         | 45 sessions      |
| Value at Risk (VaR 95%)       | 0.08%            |
| Conditional VaR (CVaR 95%)    | 0.12%            |
| Annualized Turnover (Two-Way) | 1.2x             |
| Total Executed Trades         | 13               |
| Trade Win Rate                | 92.3%            |
| Total Transaction Costs       | $2,046.94        |
| Total Short Borrow Fees       | $444.86          |

---

## 3. Monthly Return Heatmap (%)

|   year | Jan   | Feb   | Mar   | Apr   | May   | Jun   | Jul   | Aug   | Sep   | Oct   | Nov   | Dec   |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|   2020 | -     | -     | -     | -     | -     | -     | -     | -     | -     | -     | -     | 0.1%  |
|   2021 | 0.2%  | 0.2%  | 0.2%  | 0.2%  | 0.2%  | 0.0%  | 0.6%  | -0.2% | 0.5%  | 0.2%  | 0.2%  | 0.2%  |
|   2022 | 0.2%  | 0.2%  | 0.2%  | 0.2%  | 0.2%  | 0.2%  | 0.2%  | 0.2%  | 0.3%  | 0.5%  | 0.2%  | 0.4%  |
|   2023 | 0.6%  | 1.0%  | 0.1%  | 0.4%  | 0.2%  | 0.2%  | 1.0%  | -0.2% | 0.2%  | 0.2%  | 0.1%  | -     |

---

## 4. Execution Frictions & Cost Attribution

Frictional drag accounting is audited on actual executed traded notional:

| Cost Component | Dollar Impact | Effective Basis Points |
|:---|:---|:---|
| Cumulative Gross P&L | $34,692.70 | Baseline Gross |
| Modeled Slippage (5 bps/side) | $2,046.94 | 5.0 bps per side |
| Short Stock Borrow Fees | $444.86 | 50.0 bps annualized |
| **Cumulative Net Realized P&L** | **$94,638.42** | **Net of All Frictions** |
| Total Cost Drag vs Gross | 5.9% of Gross Profits | Friction Drag |

---

## 5. Visual Artifacts & Research Diagnostics

The pipeline generated the following publication-quality figures:
- `figures/equity_curve.png`: Out-of-Sample Net vs Gross Equity Curve with Cost Drag.
- `figures/drawdowns.png`: Underwater Drawdown Profile.
- `figures/rolling_sharpe.png`: 63-Day Rolling Annualized Sharpe Ratio.
- `figures/monthly_heatmap.png`: Calendar Month Return Distribution.
- `figures/exposures.png`: Gross & Net Market Neutrality Leverage Evolution.
- `figures/cost_sensitivity.png`: Sharpe Ratio & CAGR Decay as Slippage rises (0 to 20 bps).
- `figures/sample_spread_zscore.png`: Pair Spread Diagnostics & State Transitions.

---

## 6. Econometric Methodology & Multiple Comparison Control

1. **Cointegration Testing**: Engle-Granger two-step procedure evaluated against MacKinnon response-surface critical values, cross-verified with Johansen VECM trace statistics.
2. **False Discovery Rate (FDR)**: Screening $M$ candidate pairs generates substantial data snooping risk. The Benjamini-Hochberg algorithm controls FDR at $q^* = 0.10$, bounding expected false discoveries.
3. **Mean-Reversion Speed**: Spread half-life $t_{1/2} = -\ln(2)/b$ derived from AR(1) Ornstein-Uhlenbeck discretization. Pairs with $b \ge 0$ are strictly rejected.
4. **Hurst Exponent**: Variance-time scaling $\text{Var}(\Delta_\tau S) \propto \tau^{2H}$ enforces $H < 0.45$ to confirm sub-diffusive mean-reversion.

---

## 7. Institutional Risk Disclosures & Limitations

- **Microstructure Frictions**: In live markets, stocks under severe divergence can become Hard-To-Borrow (HTB), with negative rebate rates or locate failures.
- **Structural Break Risk**: Historical cointegration does not guarantee future co-movement; permanent economic divergence is capped via a strict $3.5\sigma$ statistical stop loss and 30-day time stops.
- **Survivorship Bias**: When using unadjusted current index constituents, survivorship bias is present; point-in-time constituent lists are required for survivorship-free claims.

*Report automatically generated by StatArb Research Engine.*
