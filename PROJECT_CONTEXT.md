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

**Etap 0 (architecture split) complete.** The old monolithic `quant_terminal.py`
(2711 lines: UI layout, callbacks, and math all in one file) has been split
into a layered structure with a strict one-directional dependency rule:
`ui/` may import `engine/` and `data/`; `engine/` and `data/` NEVER import
from `ui/` or from each other. `engine/` has zero Dash imports and zero
disk/network I/O -- pure functions on DataFrames/Series/arrays in, same out.
This is what makes a future ML-based `mu_i` estimator (see Section 6 roadmap)
a change confined to `engine/returns.py`, touching nothing else.

Verified behavior-preserving: callback_map keys (34/34), full layout
component-id sets (104/104), and numeric solver outputs are identical
between the old monolith and the new structure (regression-tested on the
same 6/31-ticker scenarios used throughout this project's history).

**Post-delivery bugfix pass (same day):** the numeric regression tests above
call business-logic functions directly in Python and never actually exercise
a live Dash request/callback cycle, so they could not catch missing
module-level imports in code paths only reached that way. Real-world testing
surfaced 4 such gaps, all introduced by the mechanical file-splitting itself
(a name that was "free" via shared scope in the monolith needs an explicit
import once split across files) -- not a logic change:
- `ui/components.py` was missing `import pandas as pd` (`generate_tws_matrix_styles`
  uses `pd.isna`) -- surfaced as `NameError: name 'pd' is not defined` from
  Stage 1, since that function is called from `ui/tab1_market_data.py`.
- `ui/tab1_market_data.py`, `tab2_fundamentals.py`, `tab4_rebalance.py`,
  `tab5_sandbox.py`, and `layout.py` were all missing `import dash` itself
  (they imported `dcc`/`html`/`dash_table` from it, but not the `dash` module
  used bare for `dash.no_update` / `dash.callback_context`).
- `ui/tab2_fundamentals.py` was missing the `engine.clustering` import for
  `compute_semicovariance_matrix`/`semicov_to_semicorr`, needed by
  `compute_stage3_baseline_clusters` after its move from tab1 to tab2.

Fixed and reverified via a full static AST scan across every `.py` file
(every `Name` node in `Load` context checked against imports + local
defs/params, not just the specific reported error) plus another live-app
`callback_map` check (still 34/34) -- not just a patch for the one reported
symptom. Lesson for future splits: run the AST-undefined-name scan as a
matter of course after any file-splitting refactor, before declaring it
done, rather than relying solely on functional regression tests to surface
import gaps.

```
quant-terminal/
|-- app.py                        # Entrypoint. Run this: `python3 app.py`
|-- requirements.txt
|-- PROJECT_CONTEXT.md
|-- saved_portfolios.json         # Snapshot data file (relative path, resolved from CWD)
|
|-- engine/                       # Pure math. Zero Dash imports, zero disk/network I/O.
|   |-- returns.py                #   Composite Forward Upside (mu_i) -- Section 4.2.
|   |                             #   Etap 2 (Alpha Blend) lands here, and only here.
|   |-- risk.py                   #   Estrada Sigma_eps (Tikhonov-regularized), drawdowns,
|   |                             #   CDD quantile floor, Discrete Crash-Overlap Matrix K.
|   |-- optimizer.py              #   True Two-Stage SLSQP (Section 3.4) + iterative
|   |                             #   Dynamic Singleton Split (Binding Ceiling Throttle).
|   |-- evaluation.py             #   Historical/forward backtest evaluation.
|   `-- clustering.py             #   Semi-covariance, RMT denoising, DTW + K-Medoids.
|
|-- data/                         # Persistence + market data I/O. Zero math, zero Dash.
|   |-- market_data.py            #   ALL yfinance calls -- single source of truth
|   |                             #   (previously duplicated across Stage 1 ingestion
|   |                             #   and snapshot_store.py).
|   `-- snapshot_store.py         #   Snapshot JSON CRUD + forward-return evaluation.
|
`-- ui/                           # Dash layout + callbacks. Calls engine/ and data/ only.
    |-- app_instance.py           #   The shared `app = dash.Dash(...)` object + dark CSS.
    |-- theme.py                  #   THEME dict, tab styling constants.
    |-- components.py             #   build_kpi_card, build_param_card, STAGE3/4A constants.
    |-- layout.py                 #   Assembles app.layout (all 5 tabs, unchanged structure).
    |-- tab1_market_data.py       #   Stage 1 ingestion + Stage 2 clustering.
    |-- tab2_fundamentals.py      #   Stage 3 fundamental inputs table.
    |-- tab3_tailrisk.py          #   Tail-risk analytics + Crash-Overlap visuals.
    |-- tab4_rebalance.py         #   Stage 4B True Two-Stage SLSQP solver + results.
    `-- tab5_sandbox.py           #   Forward Tracker / Sandbox.
```

**Still flat (deliberately, for now):** the 5 `ui/tabN_*.py` files mirror
today's 5-tab layout as-is -- this is NOT yet the sidebar + 4-module
structure (Overview / Research-Dossier / Portfolio Rebalance / Sandbox) from
the v2 roadmap. That reorganization is Etap 3; it reuses these same files
almost unchanged (tab4 -> Rebalance module, tab5 -> Sandbox module) rather
than requiring another rewrite of business logic.

**One deliberate non-consolidation:** `data/market_data.py` includes a new
`fetch_universe_prices()` matching Stage 1's exact fetch logic, built for
reuse by future modules (e.g. Company Dossier). Stage 1's ingestion callback
(`ui/tab1_market_data.py`) was deliberately NOT rewired to call it yet --
doing so would have collapsed two distinct error messages ("yfinance
returned nothing" vs "yfinance returned data but none of the requested
tickers resolved") into one, a user-visible behavior change outside Etap 0's
scope of "no behavior change."

---

## 5b. Etap 1 — Persistent Storage Layer (complete)

Three new/changed `data/` modules, per the v2 storage spec:

- **`data/universe_store.py`** (new) — CRUD for `storage/universe.json`, the
  master list of tracked companies (`Ticker`, `Name`, `Sector`,
  `Status: "Active_Screened"|"Watchlist"`, `Last_Updated`). `upsert_company()`
  merges rather than overwrites — a status-only update doesn't blank out a
  previously-recorded Name/Sector. Not yet wired into any UI (Etap 3/4).
- **`data/company_store.py`** (new) — CRUD for
  `storage/company_history/{TICKER}.json`, one append-only log per ticker.
  Re-recording the same `Date` updates that entry in place rather than
  duplicating it. This is the raw material for the future Company Dossier's
  target-vs-actual chart, and — much later — Etap 6's ML training data;
  starting the log now is the point, since history can't be backfilled.
  `days_since_last_update()` is already exposed for the future Overview
  module's "Data Freshness Tracker".
- **`data/snapshot_store.py`** (migrated) — was a single
  `saved_portfolios.json` array file; now one JSON file per snapshot under
  `storage/portfolio_snapshots/portfolio_{snapshot_id}.json`. Public
  function names/signatures unchanged (`save_snapshot`, `list_snapshots`,
  `get_snapshot`, `delete_snapshot`, `rename_snapshot`) except the optional
  path override parameter renamed `file_path` -> `storage_dir` (safe: no
  caller in `ui/tab4_rebalance.py` or `ui/tab5_sandbox.py` passed it
  explicitly). `evaluate_snapshot`/`compute_forward_return`/
  `holding_days_since` untouched.

**One-time migration:** `scripts/migrate_snapshots_etap1.py` moves records
out of the old `saved_portfolios.json` into the new layout, preserving the
original `snapshot_id`/`created_at` exactly (never re-stamped to "now" —
forward-tracking's holding-day count depends on the real historical creation
date). Safe by default: prints a dry-run report and writes nothing unless
`--apply` is passed; never modifies or deletes the old file under any flag.
Supports a name-substring filter (`NAME_FILTER`, currently
`"master portfolio forward"`, case-insensitive) so test snapshots don't
clutter the new storage — per an explicit instruction, only the
2026-09-01 / 31-ticker snapshot was carried over; `Test 123`,
`Test (random inputs)`, and `smiec test` were left behind in the old file
(still there as a backup, just not migrated). Pass `--keep-all` to migrate
everything instead.

Verified: full CRUD round-trip on all three stores, migration dry-run
matched the expected 1-kept/3-discarded split, migrated file byte-identical
in content to the source record (only location changed), and a full app
integration re-check after the `snapshot_store.py` rewrite (34/34 callbacks,
Tab 5 reads the migrated snapshot correctly with zero UI code changes).

---

## 5c. Etap 2 — Alpha Blend Parameter (complete)

`engine/returns.py`'s `compute_composite_upside_row()` rewritten per the
confirmed formula (2026-09-06 conversation):

$$\mu_i = \Big[(1-\alpha)\cdot U_{raw,i}\cdot\exp(-\gamma S_i) + \alpha\cdot G_{i,2Y}\Big]\cdot(1+\kappa\Delta EPS_{90d,i})\cdot\Big(1-\exp(-N_i/N_{ref})\Big)$$

Three design points confirmed before implementation:
1. **γ is coupled to the (1-α) branch, not independent of α.** At α=1.0 the
   dispersion penalty drops out entirely — correct, since S_i is a property
   of analyst *price targets* specifically, with nothing to discount once
   that branch has zero weight. (An earlier draft had γ applying globally
   regardless of α, which penalized a pure-EPS-growth estimate for
   target-price disagreement it was supposed to be ignoring — this was the
   actual bug that motivated moving γ inside the bracket.)
2. **G_2Y and ΔEPS_90d are divided by 100 inside this function only**, never
   in the Stage 3 table itself — data entry stays "55" not "0.55" (unchanged
   convention, now doubly load-bearing since skipping this scaling would
   make the EPS-growth branch outweigh the target-price branch by ~100x).
3. **A(N_i) discounts the whole bracket regardless of α, by design** — confirmed
   NOT to be coupled the way γ is. Coverage depth is treated as a general
   forecast-quality signal ("prognozy... jeżeli spółka ma większe pokrycie to
   zazwyczaj też będą bardziej jakościowe"), not specific to the target-price
   branch, so it still discounts a pure EPS-growth (α=1.0) estimate.

**Correction made during testing:** an early comment claimed α=0.5 exactly
reproduces the pre-Etap-2 formula's numbers. Verified false and fixed: the
confirmed formula also moved where `M_i` (EPS revision momentum) and `A_i`
(coverage confidence) apply — from "each modifier scoped to its own branch"
(old formula: `M_i` only multiplied the G-term, `A_i` only multiplied the
U-term) to "both apply to the whole blended bracket" (new formula) — a
structural change independent of α's value. No α reproduces the old exact
numbers; reloading a pre-Etap-2 snapshot in the Sandbox will show different
(not wrong, just recalculated under the new formula) mu_i than it originally
did. This is expected under the approved design, not a regression.

**Alpha wired everywhere parameters flow**, not just the formula:
- `ui/components.py`'s `STAGE4A_PARAMS_CONFIG` gained an `"alpha"` entry
  (range [0,1], default 0.5) — since Tab 4's param cards, `bundle_stage4a_params`,
  and `validate_param_badges` are all already loop-driven over this config,
  adding the one entry was sufficient to create the input card AND get alpha
  automatically included in the bundled `parameters` dict that
  `save_snapshot_callback` persists — **no extra code needed there**, alpha
  reaches the saved snapshot JSON for free as a consequence of that existing
  loop-driven design.
- `ui/tab3_tailrisk.py`'s diagnostic summary table gained the same `input-alpha`
  Input; its `U_adj`/`G_adj` columns were renamed `U_component`/`G_component`
  to reflect that these are no longer independently-adjusted terms (per the
  formula restructure above) but raw contributions to the pre-`M_i`/`A_i`
  bracket.
- `ui/tab4_rebalance.py`'s `run_stage4b_solver` (explicit, non-loop Input
  list) gained `Input("input-alpha", "value")` and threads it through to
  `compute_composite_upside_row`.
- `ui/tab5_sandbox.py` gained a live `slider-sb-alpha` (Tab 5 already treats
  λ/γ/κ/w_max/R_f/N_ref as live-recomputed sliders, never frozen
  snapshot-parameter reads — alpha follows the identical pattern for
  consistency, `compute_sandbox_allocation`'s `sb_alpha` parameter is never
  read from `record["parameters"]`).

**Backward compatibility for pre-Etap-2 snapshots (no "alpha" key at all):**
`compute_composite_upside_row(..., alpha=0.5)` defaults to 0.5 if the caller
passes `None` or omits it — `engine.returns.DEFAULT_ALPHA`. The one existing
migrated snapshot (`portfolio_2026-09-01_00-00.json`, Etap 1) was also
directly backfilled with `"alpha": 0.5` in its `parameters` JSON, so the
field exists everywhere rather than relying purely on a silent runtime
fallback -- per an explicit instruction to make sure alpha is present "even
in the JSON file," with 0.5 as the explicitly agreed default when unspecified.

Verified: engine-level formula tests (γ vanishes at α=1.0, A_i still
discounts at α=1.0, α∈[0,1] clipping, default=0.5), a full end-to-end
solver run showing `mu_i_map` flips correctly between α=0.0 and α=1.0 for
assets with opposite upside/growth profiles, `bundle_stage4a_params`
confirmed to include `"alpha"` automatically, and a full app integration
recheck (34/34 callbacks).

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