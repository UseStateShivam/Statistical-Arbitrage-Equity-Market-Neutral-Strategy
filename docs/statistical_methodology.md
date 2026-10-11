# Statistical Methodology: Cointegration, Econometrics & Diagnostics

## 1. Overview and Econometric Motivation

Statistical arbitrage in equity markets operates on the premise that paired securities sharing fundamental cash-flow risks, operating leverage, and sector exposure exhibit equilibrium relationships. While asset price series $P_i(t)$ individually exhibit nonstationary $I(1)$ unit root behavior, certain pairs exhibit a linear combination that is stationary $I(0)$.

This document details the econometric theory, mathematical formulas, hypothesis testing protocols, multiple comparison corrections, and diagnostic procedures implemented across the platform.

---

## 2. Cointegration Framework

### 2.1 Engle-Granger Two-Step Procedure
For two log-price series $y_t = \log(P_A(t))$ and $x_t = \log(P_B(t))$:

#### Step 1: Long-Run Static OLS Cointegrating Regression
Estimate by Ordinary Least Squares over the formation window $t \in [1, T]$:
$$y_t = \alpha + \beta x_t + \epsilon_t$$

Where:
- $\hat{\beta} = \frac{\sum (x_t - \bar{x})(y_t - \bar{y})}{\sum (x_t - \bar{x})^2}$
- $\hat{\alpha} = \bar{y} - \hat{\beta}\bar{x}$
- Residual spread: $\hat{\epsilon}_t = y_t - \hat{\alpha} - \hat{\beta}x_t$

#### Step 2: Residual Unit Root Testing
The residual series $\hat{\epsilon}_t$ is tested for stationarity using the Augmented Dickey-Fuller (ADF) regression without constant or trend (since $\hat{\epsilon}_t$ has zero sample mean by construction):
$$\Delta \hat{\epsilon}_t = \gamma \hat{\epsilon}_{t-1} + \sum_{i=1}^p \delta_i \Delta \hat{\epsilon}_{t-i} + u_t$$

- **Null Hypothesis ($H_0$)**: $\gamma = 0$ (Unit root; no cointegration).
- **Alternative Hypothesis ($H_1$)**: $\gamma < 0$ (Stationary spread; cointegration exists).
- **Test Statistic**: $t_{\gamma} = \frac{\hat{\gamma}}{\text{SE}(\hat{\gamma})}$.

#### The Econometric Caveat: MacKinnon Critical Values
Because $\hat{\epsilon}_t$ is computed from estimated parameters $(\hat{\alpha}, \hat{\beta})$ chosen to minimize residual sum of squares, ordinary Dickey-Fuller critical values underestimate the probability of rejecting $H_0$ (size distortion). We apply the MacKinnon (1991, 2010) response-surface asymptotic critical values for cointegrating residuals:
$$c_{\alpha} = \beta_{\infty} + \frac{\beta_1}{T} + \frac{\beta_2}{T^2}$$
Ensuring rigorous cointegration inference.

---

### 2.2 Johansen Cointegration Procedure
For candidate pairs, we provide Johansen's Maximum Likelihood procedure within a Vector Error Correction Model (VECM) of dimension $K=2$:

$$\Delta X_t = \mu + \Pi X_{t-1} + \sum_{i=1}^{k-1} \Gamma_i \Delta X_{t-i} + e_t$$
where $X_t = [y_t, x_t]'$, and $\Pi = \alpha \beta'$ is the matrix of long-run cointegrating vectors.

The rank $r = \text{rank}(\Pi)$ determines cointegrating relations:
- $r = 0$: No cointegrating vector; $X_t$ is a genuine $I(1)$ bivariate system.
- $r = 1$: Exactly one stationary cointegrating relationship.
- $r = 2$: Both series are already stationary $I(0)$ (degenerate).

#### Test Statistics
1. **Trace Statistic**:
   $$\lambda_{\text{trace}}(r) = -T \sum_{i=r+1}^K \ln(1 - \hat{\lambda}_i)$$
   Tests $H_0: \text{rank}(\Pi) \le r$ versus $H_1: \text{rank}(\Pi) > r$.
2. **Maximum Eigenvalue Statistic**:
   $$\lambda_{\max}(r, r+1) = -T \ln(1 - \hat{\lambda}_{r+1})$$
   Tests $H_0: \text{rank}(\Pi) = r$ versus $H_1: \text{rank}(\Pi) = r+1$.

The Johansen test treats both assets symmetrically, removing dependent variable selection bias.

---

## 3. Multiple Testing Correction: False Discovery Rate (FDR)

When screening $M = \binom{N}{2}$ candidate pairs from a universe of $N$ stocks, testing $M$ null hypotheses at nominal $\alpha = 0.05$ produces approximately $0.05 \times M$ spurious rejections by pure chance.

### 3.1 Benjamini-Hochberg (1995) Procedure
To control the False Discovery Rate $\text{FDR} \le q^*$ (default $q^* = 0.10$ or $0.05$):
1. Collect $p$-values $\{p_1, p_2, \dots, p_M\}$ from the cointegration tests across all candidate pairs.
2. Sort $p$-values in ascending order:
   $$p_{(1)} \le p_{(2)} \le \dots \le p_{(M)}$$
3. Find the maximum index $k$ such that:
   $$k = \max \left\{ i : p_{(i)} \le \frac{i}{M} q^* \right\}$$
4. Reject $H_0$ for all pairs $i \le k$.
5. Compute adjusted $q$-values (FDR-adjusted $p$-values):
   $$q_{(i)} = \min_{j \ge i} \left( \frac{M \cdot p_{(j)}}{j} \right)$$

This controls the expected proportion of false positives among discovered pairs while preserving statistical power over overly conservative Bonferroni correction.

---

## 4. Mean-Reversion Diagnostics

Cointegration establishes asymptotic stationarity, but finite-sample trading requires quantifiable mean-reversion speed and memory properties.

### 4.1 Ornstein-Uhlenbeck Half-Life Estimation
The continuous OU process discretized at $\Delta t = 1$ trading day conforms to an AR(1) representation:
$$S_t - S_{t-1} = \Delta S_t = a + b S_{t-1} + u_t$$
where:
- Estimated coefficient: $\hat{b} = \frac{\text{Cov}(\Delta S_t, S_{t-1})}{\text{Var}(S_{t-1})}$
- Mean-reversion speed: $\theta = -\ln(1 + \hat{b}) \approx -\hat{b}$ for small $|\hat{b}|$.
- **Half-Life ($t_{1/2}$)**:
  $$t_{1/2} = -\frac{\ln(2)}{\hat{b}}$$

#### Validity Filters:
1. **Stationary Condition**: $\hat{b}$ must be strictly negative ($\hat{b} < 0$). If $\hat{b} \ge 0$, $t_{1/2}$ is undefined or infinite; pair is rejected.
2. **Economic Viability Bounds**:
   - $t_{1/2} < t_{\min}$ (e.g., $< 2$ days): Mean reversion is too fast, likely microstructure bid-ask bounce; transaction costs will erode edge.
   - $t_{1/2} > t_{\max}$ (e.g., $> 63$ days): Mean reversion is too slow; capital turnover is impeded and structural break risk escalates.

---

### 4.2 Hurst Exponent Estimation
The Hurst Exponent $H \in (0, 1)$ quantifies the long-term memory of the spread:
- $H < 0.5$: Anti-persistent (mean-reverting).
- $H = 0.5$: Geometric Brownian motion / random walk.
- $H > 0.5$: Persistent (trending).

#### Implementation Methodology: Rescaled Range (R/S) & Variance-Time
For sub-series of length $\tau \in [\tau_{\min}, \tau_{\max}]$:
1. Calculate mean: $\bar{S}_\tau = \frac{1}{\tau} \sum_{i=1}^\tau S_i$
2. Mean-adjusted cumulative deviations: $X(k, \tau) = \sum_{i=1}^k (S_i - \bar{S}_\tau)$
3. Range: $R(\tau) = \max_k X(k, \tau) - \min_k X(k, \tau)$
4. Standard deviation: $S(\tau) = \sqrt{\frac{1}{\tau} \sum_{i=1}^\tau (S_i - \bar{S}_\tau)^2}$
5. Rescaled range: $(R/S)_\tau = \frac{R(\tau)}{S(\tau)}$
6. Regress $\log((R/S)_\tau)$ on $\log(\tau)$:
   $$\log((R/S)_\tau) = c + H \log(\tau) + \nu_\tau$$

**Filter Rule**: Require $H \le H_{\max}$ (default $0.45$). Any pair with $H \ge 0.50$ is rejected.

---

## 5. Structural Break & Stability Testing

A pair may appear cointegrated in-sample but suffer a structural break (e.g., M&A announcement, divergence in debt structure, technological disruption).

### 5.1 Chow Breakpoint Test
Given a candidate split point $T^*$:
- Unrestricted model: fit OLS separately on $[1, T^*]$ and $[T^*+1, T]$ with SSR $S_1, S_2$.
- Restricted model: fit OLS on pooled $[1, T]$ with SSR $S_c$.
- Chow test statistic:
  $$F = \frac{(S_c - (S_1 + S_2)) / k}{(S_1 + S_2) / (T - 2k)} \sim F(k, T - 2k)$$
Where $k=2$ (intercept and hedge ratio). A statistically significant $F$ indicates a structural break.

### 5.2 Rolling Hedge Ratio Stability
Hedge ratio stability is evaluated over rolling 63-day sub-windows. If the coefficient of variation $\text{CV}(\hat{\beta}) = \frac{\sigma_\beta}{\bar{\beta}}$ exceeds a predefined threshold (default $0.25$), the pair relationship is classified as unstable.
