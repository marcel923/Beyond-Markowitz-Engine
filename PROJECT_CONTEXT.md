# Beyond-Markowitz: Quantitative Growth & Tail-Risk Engine

**Author:** Marcel Zenderowski  
**Affiliation:** Incoming Undergraduate, SGH Warsaw School of Economics  
**Theoretical Foundation:** *Beyond-Markowitz: A Working Framework for Portfolio Optimization via Estrada Downside Semi-Covariance, Discrete Crash-Overlap Matrices, and True Two-Stage SLSQP Optimization*  
**Status:** Active Empirical Forward-Testing & Production Calibration  

---

## 1. Problem Formulation and Theoretical Motivation

Classical Mean-Variance Optimization (Markowitz, 1952) operates on assumptions that fail under empirical equity market conditions:
* **Symmetric Volatility Penalization:** Standard variance ($\sigma^2$) penalizes positive return dispersion identically to downside losses. Rational capital allocation demands penalizing downside volatility relative to a target floor ($\text{MAR} = 0$) while leaving upside deviations unpenalized.
* **Passive Inflow Distortions:** The proliferation of passive indexation (e.g., S&P 500 and Nasdaq-100 ETF vehicles) inflates cross-asset Pearson correlation coefficients, compromising naive diversification structures.
* **Tail-Risk Blindness & Temporal Decoupling:** Standard Sharpe and Sortino ratios evaluate return moments without capturing path-dependent structural drawdown overlap. Assets with identical semi-variances may crash at identical historical dates, compounding portfolio tail risk, or crash asynchronously, providing organic structural diversification.

The Beyond-Markowitz framework implements a quantitative pipeline designed to:
1. Estimate downside co-movement exclusively during negative return intervals via Estrada's Semi-Covariance Matrix with spectral Tikhonov regularization.
2. Formulate asset forward returns ($\mu_i$) using fundamental growth vectors, consensus price targets discounted by forecast spread, and 90-day analyst revision momentum.
3. Quantify joint tail risks through the **Discrete Crash-Overlap Matrix** ($\mathbf{K} = \mathbf{J} \odot \mathbf{S}$), combining temporal drawdown overlap ($\mathbf{J}$) with severe drawdown magnitude ($\mathbf{S}$).
4. Allocate capital via a **True Two-Stage SLSQP Model**: Stage 1 solves intra-cluster optimal weights maximizing cluster-level Tail-Penalized Sortino ($TPS_{C_k}$), while Stage 2 solves global inter-cluster weights subject to hard concentration ceilings ($w_i \le w_{\max}$).
5. Eliminate look-ahead and backtesting biases using a live out-of-sample forward tracking architecture.

---

## 2. Investment Philosophy: Pure Hyper-Growth & Structural Risk-Bounding

The framework rejects broad-market beta-smoothing in favor of **100% Fully-Invested, Unleveraged Hyper-Growth Allocation**:
* **Secular Hyper-Growth Universe ($G_{2Y} \ge 20\%$):** Candidates are strictly pre-filtered to exclude mature/low-growth equities. Any asset with forward revenue/EPS growth expectations below 20% is eliminated a priori.
* **Momentum Continuity via Dual Expected Return ($\mu_i$):** By balancing analyst price targets ($U_{\text{adj}}$) with fundamental growth momentum ($G_{\text{adj}}$), the engine prevents premature liquidation of momentum winners whose spot price has temporarily exceeded sell-side target prices ($P_0 > T_i$).
* **Aggressive Hurdle Rate Formulation ($R_f$ as a Quality Filter):** The risk-free benchmark $R_f$ is treated as an aggressive hurdle rate (calibrated up to $10\%\text{--}15\%$) to isolate hyper-growth compounding engines from mediocre performers.
* **Full Equity Exposure (100% Equity, Zero Cash/Leverage):** The engine operates with zero financial leverage ($0 \le w_i \le w_{\max}$, $\sum w_i = 1.0$) and no cash drag. Short-term drawdowns (liquidity panics, flash crashes) are treated as transient noise, provided underlying earnings momentum ($\Delta\text{EPS}_{90d} \ge 0$) remains intact.
* **Human-in-the-Loop Cluster Discretion:** Unsupervised clustering algorithms (DTW, Ward, Semi-Covariance) provide quantitative baseline groupings. To address structural business-model inflections, the architecture maintains interactive, manual cluster override capabilities.

---

## 3. Mathematical Framework

### 3.1 Composite Forward Upside ($\mu_i$)
Asset expected returns are computed as a weighted combination of target consensus projections and fundamental earnings growth:

$$\mu_i = w_1 \cdot U_{\text{adj}, i} + w_2 \cdot G_{\text{adj}, i} \quad \text{where } w_1 = w_2 = 0.5$$

#### A. Target Price Consensus Component ($U_{\text{adj}, i}$):
$$U_{\text{raw}, i} = \frac{T_i - P_{0, i}}{P_{0, i}}$$

$$S_i = \frac{T_{\text{high}, i} - T_{\text{low}, i}}{T_i}$$

$$A_i = 1 - \exp\left(-\frac{N_i}{N_{\text{ref}}}\right)$$

$$U_{\text{adj}, i} = U_{\text{raw}, i} \cdot \exp(-\gamma \cdot S_i) \cdot A_i$$

where $T_i$ is consensus target price, $P_{0,i}$ is spot price, $S_i$ measures analyst target spread uncertainty, and $A_i$ represents analyst coverage confidence scaling.

#### B. Fundamental EPS Momentum Component ($G_{\text{adj}, i}$):
$$M_i = 1 + \kappa \cdot \left(\frac{\Delta\text{EPS}_{90d, i}}{100}\right)$$

$$G_{\text{adj}, i} = \left(\frac{G_{i, 2Y}}{100}\right) \cdot M_i$$

---

### 3.2 Regularized Estrada Downside Semi-Covariance Matrix ($\Sigma_{\epsilon}$)
Given daily returns $r_{i,t}$ over $T$ trading observations and a minimum acceptable return floor $\text{MAR} = 0$:

$$\Sigma_{\text{downside}, i, j} = \left( \frac{1}{T} \sum_{t=1}^T \min(r_{i,t}, 0) \cdot \min(r_{j,t}, 0) \right) \cdot 252$$

To guarantee strict positive semi-definiteness ($PSD$) and eliminate singularity during non-convex optimization, Tikhonov spectral regularization is applied:

$$\Sigma_{\epsilon} = \Sigma_{\text{downside}} + \epsilon_{\text{ridge}} \cdot I_N, \quad \text{where } \epsilon_{\text{ridge}} = 10^{-8}$$

---

### 3.3 Discrete Crash-Overlap Matrix ($\mathbf{K} = \mathbf{J} \odot \mathbf{S}$)

Rather than relying on continuous daily underwater trajectories, tail-risk coupling is modeled as a discrete joint crash-event matrix:

#### A. Temporal Crash-Overlap Matrix ($\mathbf{J}$):
For each asset $i$, define the drawdown state indicator:
$$I_{i,t} = \mathbb{I}(DD_{i,t} \le -CDD_{0.10, i})$$
The pairwise temporal overlap coefficient is formulated via the Jaccard similarity index across trading sessions $T$:
$$\mathbf{J}_{i,j} = \frac{\sum_{t=1}^T (I_{i,t} \land I_{j,t})}{\sum_{t=1}^T (I_{i,t} \lor I_{j,t})}$$
* $\mathbf{J}_{i,j} \to 1.0$: Identical historical crash timing (high tail contagion).
* $\mathbf{J}_{i,j} = 0.0$: Asynchronous corrections (perfect tail diversification).

#### B. Severity Magnitude Matrix ($\mathbf{S}$):
Given the 10th percentile empirical drawdown vector $v = [CDD_{0.10, 1}, \dots, CDD_{0.10, N}]^T$:
$$\mathbf{S} = v \cdot v^T, \quad \mathbf{S}_{i,j} = CDD_{0.10, i} \cdot CDD_{0.10, j}$$

#### C. Composite Crash-Risk Matrix ($\mathbf{K}$):
By the Schur Product Theorem, the element-wise Hadamard product yields a strictly symmetric positive semi-definite risk matrix:
$$\mathbf{K} = \mathbf{J} \odot \mathbf{S}, \quad K_{i,j} = \mathbf{J}_{i,j} \cdot \mathbf{S}_{i,j}$$

For any weight vector $w$, the portfolio tail penalty is:
$$P_P(w) = \exp\left( \lambda \cdot \sqrt{w^T \cdot \mathbf{K} \cdot w} \right)$$

---

### 3.4 True Two-Stage SLSQP Portfolio Optimization Engine

$TPS$ is treated as an optimization objective function over weight vectors rather than an isolated asset metric.

#### Stage 1: Intra-Cluster SLSQP Optimization
For each cluster $C_k$ containing sub-universe indices $i \in C_k$, solve:
$$\max_{g \in \mathbb{R}^{n_k}} \quad TPS_{C_k}(g) = \frac{g^T \mu_{C_k} - R_f}{\sqrt{g^T \Sigma_{\epsilon, C_k} g} \cdot \exp\left(\lambda \sqrt{g^T \mathbf{K}_{C_k} g}\right) + \epsilon}$$
$$\text{subject to} \quad \sum_{i \in C_k} g_i = 1.0, \quad 0 \le g_i \le 1.0$$
Yields optimal intra-cluster weights $g_k^*$ and maximized cluster performance $TPS_{C_k}^*$.

#### Stage 2: Inter-Cluster SLSQP Optimization
With macro-cluster profiles defined by $(g_k^*, \mu_{C_k}^*, \Sigma_{\epsilon, C_k}, \mathbf{K}_{C_k})$, solve for cluster weights $W = [W_{C_1}, \dots, W_{C_M}]^T$:
$$\max_W \quad TPS_P(W) = \frac{w_{\text{final}}^T \mu - R_f}{\sqrt{w_{\text{final}}^T \Sigma_{\epsilon} w_{\text{final}}} \cdot \exp\left(\lambda \sqrt{w_{\text{final}}^T \mathbf{K} w_{\text{final}}}\right) + \epsilon}$$
$$\text{subject to} \quad \sum_{k=1}^M W_{C_k} = 1.0, \quad W_{C_k} \ge 0$$
$$\text{and global box constraints:} \quad w_{\text{final}, i} = W_{C_k} \cdot g_{k, i}^* \le w_{\max} \quad (\forall i)$$

---

## 4. Execution Protocol & Rebalancing Dynamics

1. **Monthly Periodic Re-Optimization (Standard Cadence):**
   * Re-estimation of all matrices ($\Sigma_{\epsilon}, \mathbf{K}$) and forward parameters ($\mu_i$) every 21 trading sessions.
   * Capital redistribution across the active constituent universe based on updated SLSQP global weights.
2. **Event-Driven Asymmetric Fast-Exit Trigger:**
   * If an active constituent undergoes sudden negative consensus revisions ($\Delta\text{EPS}_{90d} < 0$), the position is liquidated immediately out of cycle.
   * Capital is dynamically reallocated pro-rata across remaining active positions with positive momentum scores.
3. **Data Integrity Feasibility Floor ($T_{\min}$):**
   * Candidates must exhibit a minimum continuous trading history of $T_{\min} \ge 756\text{ trading sessions}$ ($\sim 3\text{ years}$).

---

## 5. System Architecture & Module Specifications

quant-terminal/
|-- quant_terminal.py          # Dash application: UI layout, state engine, dark CSS, callbacks
|-- tps_solver.py              # Mathematical engine: Estrada matrix, Crash matrix K, Stage 1/2 SLSQP solvers
|-- snapshot_store.py          # Persistence engine: Atomic JSON I/O, live yfinance sync, tracking math
|-- saved_portfolios.json      # Structured persistence layer for out-of-sample portfolio states
|-- requirements.txt           # Environment dependencies
`-- PROJECT_CONTEXT.md         # Technical architecture documentation

---

## 6. Implementation Status & Development Roadmap

### Currently Implemented in Codebase:
* [x] Ingestion pipeline with automated yfinance handling (MultiIndex defensive normalization).
* [x] Basic Estrada semi-covariance matrix calculation ($\text{MAR} = 0$) on daily log returns.
* [x] Continuous 10th percentile empirical drawdown derivation ($CDD_{0.10}$).
* [x] Composite Forward Upside formula integrating analyst target spreads and EPS momentum.
* [x] Interactive Plotly Dash dashboard with 3D optimization surface and offcanvas raw computation inspector.
* [x] Out-of-sample forward tracking module logging benchmark-relative performance against live equity data.
* [x] **Fix Dark Theme UI Controls:** Pinned `dash>=3.3,<4.0` in `requirements.txt` to preserve React DOM wrappers; fixed main tab bar styling via `style`/`selected_style` props.
* [x] **Spectral Tikhonov Regularization:** Implemented in `compute_estrada_matrix()` (`tps_solver.py`): `RIDGE_EPSILON = 1e-8` added to the diagonal of the annualized `Sigma_downside` (`apply_ridge=True` by default, all call sites unaffected). Note: `Sigma_downside` is a Gram matrix and thus already PSD by construction — the ridge's real job is guarding against near-singularity/ill-conditioning (e.g. two names in the same cluster with near-collinear return series), not "making it PSD".
* [x] **Discrete Crash-Overlap Matrix ($\mathbf{K} = \mathbf{J} \odot \mathbf{S}$):** Implemented as `compute_drawdown_series()`, `compute_cdd_quantile_vec()`, and `compute_crash_overlap_matrix()` in `tps_solver.py` (CDD/drawdown math moved here from `quant_terminal.py` for module-split consistency, per Section 5). **Design decision — pairwise date intersection ("Option A"):** the universe mixes exchanges with different trading calendars (NYSE/NASDAQ, KRX `005930.KS`/`000660.KS`, ASX `LYC.AX`/`SIG.AX`, LSE `ANTO.L`) plus potentially short-history names, so $DD_{i,t}$ is computed on each asset's own full price history (its `cummax` is never truncated by another asset's gaps), but $\mathbf{J}_{i,j}$ for each pair is summed only over the trading days where BOTH assets have a real price (`valid_price[i] & valid_price[j]`) — not over a single shared universe-wide window. A universe-wide window would either miscount a foreign-exchange/short-history name's non-trading days as "not crashing" (deflating $\mathbf{J}$) or force truncating every asset down to the youngest listing (data loss adjacent to what the $T_{\min}$ floor in Section 4 is meant to prevent). $O(N^2)$ pairwise loop, negligible at this project's ticker-count scale. Wired into UI: new `dcc.Store(id="store-crash-matrices")`, populated by a dedicated callback off `store-raw-close`; Tab 3 shows a $\mathbf{J}$ heatmap, a $\mathbf{K}$ heatmap, and a Crash-Risk Contribution ranking table (row-sum of $\mathbf{K}$ per ticker) — inspector was intentionally skipped, everything lives directly in Tab 3 per request. **Now consumed by the solver** — see the next item; `compute_tail_penalty()`/$P_i$ (Z-score-based) is fully decoupled from allocation, kept only as a Tab 3 diagnostic (see below). Incidental fix made while touching this code: `compute_tail_risk_metrics()`'s old `"dates"` field assumed every ticker shared the first ticker's date index (false for ragged/multi-exchange histories, could misalign the underwater chart's x-axis for any ticker other than the first) — replaced with a per-ticker `"dd_dates"` dict.
* [x] **True Two-Stage SLSQP Solver Refactor:** `tps_solver.py`'s old heuristic (`_power_weighted_normalize`, `run_two_stage_allocation`) and old single-pass linear-penalty refinement (`optimize_portfolio_slsqp`, `_portfolio_tps_neg`) are fully removed. Replaced by `run_stage1_intra_cluster_slsqp()` (per-cluster SLSQP maximizing $TPS_{C_k}(g) = (g^T\mu_{C_k}-R_f)/(\sqrt{g^T\Sigma_{\epsilon,C_k}g}\cdot\exp(\lambda\sqrt{g^TK_{C_k}g})+\epsilon)$, $\sum g_i=1$, $0\le g_i\le1$), `run_stage2_inter_cluster_slsqp()` (global SLSQP over cluster weights $W$, reconstructing $w_{final,i}=W_{C_k}\cdot g^*_{k,i}$ against the FULL $N\times N$ $\Sigma_\epsilon$/$\mathbf{K}$ — cross-cluster terms are not discarded just because the decision variable is smaller; $w_{max}$ enforced as a per-cluster bound $W_{C_k}\le\min(1,w_{max}/\max_i g^*_{k,i})$, not an inequality constraint, for solver stability), and `run_true_two_stage_optimization()` (thin Stage1→Stage2 wrapper). $P_i$ no longer reaches the solver at all — objective depends only on $\mu$, $\Sigma_\epsilon$, $\mathbf{K}$. Side benefit: the old "equal-weight fallback when every $TPS_i\le0$" hack (needed because `x^{power}` is degenerate for non-positive `x`) no longer exists — SLSQP's objective is smooth regardless of the sign of $\mu_i$.
  **Dynamic Singleton Split — Iterative Multi-Pass with Binding Ceiling Throttle (final design, evolved through production testing):** Stage 1 has no knowledge of $w_{max}$ (per spec, only $[0,1]$ bounds), so it can legitimately concentrate a cluster's $g_i$ past what Stage 2's single remaining degree of freedom ($W_{C_k}$) can then repair.
  *Evolution, in order:*
  (1) **Two-Pass, infeasibility-only trigger** (first version): split only fired when Stage 2 raised `ValueError`. Production testing (real screenshots comparing a 3-cluster vs. a manually-forced 1-cluster run) showed this missed a real "silent throttling" failure mode: a dominant ticker's $g_i>w_{max}$ caps its whole cluster's $W_{C_k}$ below 1.0 even when OTHER clusters have enough slack to keep the overall optimization formally feasible — so it converges without ever raising, quietly squeezing that dominant ticker's cluster-mates with no visible signal. Also, rigid two passes couldn't handle cascading dominance (removing one dominant member concentrating the remainder enough to reveal a new one) — verified against the user's own reproduced case ($w_{max}=0.30$, sum of implied caps $=0.9145$) before fixing it.
  (2) **Iterative Multi-Pass with a naive `g_i > w_max` trigger** (second version): replaced the two fixed passes with a `while` loop (bounded by `max_passes`, default = asset count), checking `g_i > w_max` on EVERY pass regardless of feasibility. This correctly caught the silent-throttling case, but production testing immediately surfaced a new, more serious problem: in a 2-member cluster, the "dominant" half is almost always $>50\%$, so at any realistic $w_{max}$ (0.30–0.35) this criterion fires on nearly every small cluster whether or not anything was actually being constrained — e.g. a 3-cluster, 6-ticker test at $w_{max}=0.35$ atomized 5 of 6 tickers into singletons across cascading passes, destroying the clustering structure for no real gain (the "cluster" concept degenerates toward a 1-stage solver).
  (3) **Binding Ceiling Throttle (current, final version):** a ticker is only promoted if BOTH hold: (a) $g_i>w_{max}$ (structural precondition, unchanged), AND (b) the cluster's SOLVED $W_{C_k}$ from Stage 2 is actually pinned at its own implied ceiling — $W_{C_k}\ge w_{max}/\max_i g_i-\text{tol}$ — or the whole pass was infeasible (no $W_{C_k}$ to check, so infeasibility itself counts as binding by definition). This distinguishes "this ticker COULD dominate if given the chance" (condition (a) alone, harmless if the cluster never gets enough portfolio-level demand to bump into its own bound) from "this ticker IS ACTIVELY blocking capital Stage 2 wanted to allocate" (both conditions, genuine throttling). Re-running the same 3-cluster/6-ticker test that atomized 5 of 6 tickers under version (2) now promotes only 1 (the one genuinely pinned at its ceiling) — the rest keep their original cluster structure.
  Only the cluster's single most-dominant member is ever flagged per cluster per pass (`_identify_throttling_tickers` in `tps_solver.py`); `_build_split_clusters` carves it into its own `SINGLETON_<ticker>` cluster and the loop repeats. `run_optimization_with_singleton_split()` returns a full `"history"` (one entry per pass: cluster structure, weights, $g$, feasibility, which ticker(s) were newly promoted after that pass) rather than a fixed `pass1`/`pass2` pair, plus `"final"`, `"split_triggered"`, `"promoted_tickers"`, `"final_clusters_dict"`, `"n_passes"`. This is the **top-level entry point** both the main Stage 4B solver and the Tab 5 sandbox call — never `run_true_two_stage_optimization` directly.
  **UI consequences:** `gamma_intra`/`eta_inter` sliders removed entirely from `STAGE4A_PARAMS_CONFIG` and Tab 5 (dead parameters — a real SLSQP has no "concentration power" knob). $\lambda$ rescaled from `[0.40, 1.20]` to `[0.5, 15.0]` (step 0.5, default 3.0): the new penalty is $\exp(\lambda\sqrt{w^TKw})$ where $\sqrt{w^TKw}\approx0.05\text{–}0.20$ empirically, an order of magnitude smaller than the old Z-score exponent range, so $\lambda$ needed to widen to have a comparable effect. $w_{max}$ widened to `[0.10, 0.50]` (step 0.05, default 0.30). **Legacy $P_i$ diagnostic decoupled from `input-lambda`:** Tab 3's `render_stage4a_summary_table`/`render_stage4a_underwater_chart` (the old $P_i=\exp(\lambda\cdot Z)$ display) no longer read the (now much larger) solver $\lambda$ — doing so would blow up to nonsensical values (e.g. $\exp(15\times2)$ at a $Z=+2$ outlier) purely because the two formulas happened to share a variable name. Fixed at a new `LEGACY_DIAGNOSTIC_LAMBDA = 0.60` constant instead (the pre-refactor default), fully independent of the solver from now on. Standalone Asset TPS (`compute_asset_tps`, the metric formerly used to *drive* the old heuristic) is now computed as a pure post-hoc diagnostic in the weights table only — relabeled "Standalone Asset TPS (diagnostyka, nie steruje alokacją)" — explicitly disconnected from allocation.
  **Tab 4 visibility:** new `panel-singleton-split` renders an alert (`"⚡ Auto-promocja do singletona: [tickers] ... w_max=X%"`) plus a grouped bar chart comparing Pass 1 vs Pass 2 weights whenever a split fires, hidden otherwise. Note: Pass 1 has no literal portfolio weights when infeasible (Stage 2 aborted before producing any), so "before" in that chart is an uncapped baseline — Stage 1's frozen $g^*$ run through Stage 2 again with $w_{max}=1.0$ (i.e. "what the model wanted before hitting the wall"), not Pass 1's (nonexistent) actual output.
* [x] **Unify Tab 5 Solver Execution:** `compute_sandbox_allocation()` in `quant_terminal.py` now calls `ts.run_optimization_with_singleton_split()` — the exact same entry point as the main Stage 4B solver, no separate sandbox-only code path. Price window widened from ~2Y (730 calendar days, `.tail(504)`) to **5Y** (1825 calendar days, `.tail(1260)`) for both the sandbox and confirmed already-5Y for the main pipeline's Stage 1 ingestion (`yf.download(..., period="5y", ...)` was already correct there) — so $\Sigma_\epsilon$ and $\mathbf{K}$ see multi-year cycles and deep historical corrections in both places, not just the last two years. Raw (NaN-preserving) prices are now passed through to `compute_crash_overlap_matrix` unchanged (previously the caller pre-applied `.dropna(how="any")` before the sandbox function even ran, which would have silently defeated Option A's per-pair date-intersection handling for K); `.dropna()` is still applied locally, only where a clean common date index is actually required (the `Sigma_\epsilon` return-matrix step). `cap_weights_iteratively()` (the old water-filling cap) is kept only as a defensive safety net for the equal-weight fallback paths (insufficient data) — on the solver path, $w_{max}$ is already enforced by the SLSQP bounds + Singleton Split, so it's a no-op there in practice, not the primary mechanism anymore.

### Planned Implementation & Development Backlog (To-Do):
* [ ] **Multi-Horizon Drawdown Persistence Filter:** Implement $\Delta TR_i$ evaluation ($5\text{Y}$ vs. $3\text{Y}$) to discount historical-only drawdown anomalies for restructuring equities.
* [ ] **Automated Fundamental Ingestion:** Transition Stage 3 inputs from manual tabular entry to programmatic API scraping (e.g., FMP or Yahoo Quote Summary).
* [ ] **Dynamic Rebalancing Simulator:** Implement periodic rebalancing backtests with execution friction (bid-ask spread and slippage modelling).
* [ ] **Asymmetric Options Tail-Hedge Overlay:** Build a systemic collar/put-spread module to evaluate hedging overlays for tail-risk truncation once portfolio scale permits.

---

## 7. Execution Instructions

### Prerequisites:
* Python `>= 3.10`
* Dependencies listed in `requirements.txt`: