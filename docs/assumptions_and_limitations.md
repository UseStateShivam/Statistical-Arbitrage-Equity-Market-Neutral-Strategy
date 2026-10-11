# Assumptions, Limitations & Institutional Risk Disclosures

## 1. Introduction

Quantitative research that fails to document its implicit assumptions and real-world execution frictions is prone to severe backtest overfitting and live deployment failure. This document transparently articulates all economic, econometric, market microstructure, and accounting assumptions underpinning this research platform, alongside explicit risk disclosures.

---

## 2. Information Timing & Prevention of Look-Ahead Bias

### 2.1 Causal Execution Protocol
- **Signal Observation**: End of trading day $t$ close ($16:00$ EST). All indicators—including log-prices, hedge ratios, rolling spread means $\mu_{t-1}$, spread standard deviations $\sigma_{t-1}$, and Z-scores $Z_t$—incorporate data strictly up to day $t$ close.
- **Order Generation**: Orders are calculated post-market at day $t$ close.
- **Execution Fill**: Orders are filled at day $t+1$ market open ($09:30$ EST) or open auction price.
- **Prohibition of Same-Close Execution**: Never execute at day $t$ close on a signal generated from day $t$ close prices.

### 2.2 Rolling Parameter Isolation
- Formation window parameters (e.g., cointegrating vectors $(\hat{\alpha}, \hat{\beta})$, half-life, Hurst exponent) are estimated over historical windows $[T-W_F, T]$ and frozen during out-of-sample forward trading $[T+1, T+W_T]$.
- Future observations are never accessed during retrospective backtesting. Unit tests explicitly enforce this invariant via perturbation tests.

---

## 3. Short Selling & Borrow Assumptions

### 3.1 Borrow Availability
- Pairs trading is fundamentally bilateral: every trade requires shorting one leg.
- In simulated backtests, we assume all large-cap liquid equities ($P \ge \$5$, $\text{ADTV} \ge \$5\text{M}$) are Easy-To-Borrow (ETB).
- **Limitation**: In live equity markets, stocks experiencing idiosyncratic divergence frequently become Hard-To-Borrow (HTB) with negative rebate rates, sudden borrow recalls, or locate failures precisely when the arbitrage opportunity is largest.

### 3.2 Borrow Financing Cost
- Baseline modeling incorporates a continuous borrow fee of **50 basis points per annum** ($0.50\%$ annualized, or $0.0050 / 252$ per calendar day) charged on overnight short market value.
- Short sale proceeds are assumed to earn the risk-free cash rate less broker haircut.

---

## 4. Execution Friction & Transaction Costs

### 4.1 Slippage Model
- **Baseline Requirement**: **5 basis points (0.05%) per traded side** on all executed notional.
- For a trade rebalancing $\$100,000$ of equity notional, a $\$50$ fee is deducted immediately from cash.
- **Slippage Sensitivity**: Tested under zero-cost ($0$ bps), baseline ($5$ bps), and stressed ($10$ bps, $15$ bps, $20$ bps) scenarios to evaluate strategy edge resilience.

### 4.2 Liquidity & Market Impact
- Sizing enforces that no single order exceeds $1.0\%$ of the asset's trailing 20-day Average Daily Volume.
- Non-linear square-root market impact (Almgren-Chriss formulation) is documented as an advanced research extension.

---

## 5. Econometric & Statistical Limitations

### 5.1 Nonstationarity & Structural Break Risk
- Cointegration is an asymptotic property estimated over historical samples. Long-run economic ties can permanently break due to:
  - Corporate restructuring, mergers, spin-offs, or leveraged buyouts.
  - Technological disruption (e.g., divergence between legacy automakers and EV manufacturers).
  - Credit degradation and capital structure divergences.
- Stop-loss rules ($|Z_t| \ge 3.5$) and maximum holding horizons are mandatory safeguards against diverging non-cointegrated spreads.

### 5.2 Multiple Hypothesis Testing & Selection Bias
- Screening hundreds of pairwise combinations across equity universes creates significant data-snooping risk.
- While the Benjamini-Hochberg FDR control bounds expected false discoveries under independence or positive dependence, equity residuals within the same sector share latent factor exposures, violating strict independence.

---

## 6. Survivorship & Universe Limitations

### 6.1 Point-in-Time Universe
- If historical point-in-time constituent datasets with delisted assets are unavailable, universe selection on current index constituents introduces **survivorship bias**.
- Stocks that went bankrupt, were acquired at distressed valuations, or were delisted during the historical window are omitted from survivorship-unadjusted datasets.
- The platform explicitly flags data source survivorship status and avoids claiming survivorship-free status unless verified point-in-time membership datasets are supplied.

---

## 7. Portfolio Accounting & Margin Treatment

### 7.1 Capital & Leverage Definition
- Portfolio Equity ($E_t$): $\text{Cash}_t + \sum_i \text{Holdings}_{i, t} \times P_{i, t}$.
- Gross Exposure: $\frac{\sum_i |\text{Market Value}_{i, t}|}{E_t}$.
- Net Exposure: $\frac{\sum_i \text{Market Value}_{i, t}}{E_t}$.
- Return Denominator: Daily portfolio returns are computed with respect to total portfolio net equity: $R_t = \frac{E_t - E_{t-1}}{E_{t-1}}$.
- Margin calls and liquidation events are simulated if portfolio equity drops below maintenance margin thresholds ($30\%$ of gross exposure).
