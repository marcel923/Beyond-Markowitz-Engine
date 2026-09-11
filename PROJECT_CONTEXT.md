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

## 5d. Visual Design System — "Institutional, Softened" (complete)

Replaced the original vibrant-SaaS palette (8-color rainbow CLUSTER_PALETTE,
purple-and-orange used interchangeably for unrelated concepts) after a
design discussion (2026-09-07) comparing three references: a vibrant neon
"FX Dashboard" concept (rejected — hard to execute consistently in Dash/
Plotly, risks looking amateurish rather than premium), a Tempo-style
consumer SaaS dashboard (rejected on a second pass — too soft/rounded,
"looks like an iPhone app"), and Interactive Brokers' desktop terminal
(the anchor reference — dense, grid-ruled, functional color only). Iterated
through a too-severe "raw trading workstation" pass (all-monospace,
zero-radius, pure black) before settling on the current middle ground.

**Final direction, `ui/theme.py`:**
- Navy-tinted charcoal (`#0B0E14`/`#10141C`), not pure black — the "granatowy"
  (navy) admixture that reads as IBKR's actual desktop app rather than a
  bare terminal.
- Normal sans-serif everywhere, NOT monospace — numeric alignment achieved
  via `fontVariantNumeric: "tabular-nums"` on values/table cells instead of
  a typeface swap. A full-monospace first pass read as "hacker terminal",
  not "professional fintech".
- Small 4-6px radius on panels/inputs/badges (not 0, not the old 16px) —
  enough softening to avoid "cut with a knife", not enough to read consumer-app.
- ONE functional accent (`THEME["accent"]`, institutional blue `#5B9FEF`,
  renamed from `"purple"` — keeping a key literally named "purple" while it
  held a blue value would have been actively misleading in the ~26 call
  sites that reference it).
- Semantic color split: `THEME["pos"]` (green, gains) / `THEME["neg"]` (red,
  losses) / `THEME["warn"]` (amber, warnings/attention/capped-flags) replace
  the old overloaded pattern where a hardcoded `"#00E5A0"` neon green
  (scattered outside THEME entirely) paired inconsistently with
  `THEME["orange"]` for both genuine losses AND unrelated warnings/errors.
  Of ~54 `THEME["orange"]` call sites audited, only ~9 were genuine
  numeric-loss cases and migrated to `"neg"`; the remaining ~40 (error
  messages, capped/promoted-singleton flags, chart threshold annotations)
  were already semantically "attention/warning", so they need no code
  change — `THEME["orange"]` is kept as a deprecated alias pointing at the
  same value as `"warn"`, specifically so unmigrated call sites keep
  rendering correctly rather than `KeyError`ing.
- `CHART_COLORS`/`CLUSTER_PALETTE` reduced from 8 vivid hues to 6 muted ones
  — cluster differentiation no longer competes visually with the semantic
  pos/neg/warn colors used elsewhere.
- `ui/components.py` gained three shared DataTable style helpers
  (`datatable_style_header/cell/data`) and a row-striping rule
  (`datatable_row_alt_rule`) — applied to all 7 `dash_table.DataTable`
  definitions across `layout.py`/`tab1`/`tab3`/`tab5` (previously each had
  independently hardcoded, inconsistent hex values). `build_kpi_card`/
  `build_param_card` updated to the new radius/padding/tabular-nums.

**A real bug found and fixed during this pass, worth remembering:** an
early attempt converted `ui/app_instance.py`'s injected `dark_css` /
`app.index_string` CSS blocks into f-strings interpolating `THEME` directly
(for single-source-of-truth). This passed `py_compile` but broke at actual
import time — `NameError: name 'background' is not defined`. Cause: literal
CSS rule braces (`.Select-control { ... }`) collide with Python's f-string
`{expr}` syntax; every literal brace in an f-string must be escaped as
`{{`/`}}`, easy to miss across dozens of CSS rules, and `py_compile` only
validates bytecode generation, not that every `{...}` resolves to a valid
expression at *runtime* string-formatting time. Fixed by reverting to plain
(non-f) triple-quoted strings with hardcoded hex values matching the new
palette — sacrifices single-source-of-truth for this one file, documented
inline so a future palette change remembers to update it manually. Lesson:
after any f-string/`.format()` conversion of a block containing literal
braces, actually import and inspect the runtime value — a clean
`py_compile` is not sufficient proof of correctness for that class of bug.

Verified: full app integration recheck (34/34 callbacks) after every edit
batch, a solver regression run confirming the redesign touched zero
business logic (identical weights to pre-redesign runs on the same inputs),
and a runtime import check specifically for the f-string bug class above.

---

## 5e. Design Correction — Vivid Semantic Colors + Connected Panels (complete)

Follow-up to Section 5d after real-world screenshots surfaced two problems
with that first pass:

1. **Colors read as washed out ("jak dla daltonisty").** The mistake:
   conflating "soften the structure" (radius, spacing, typography) with
   "soften the colors" -- these are different axes, and only the first
   should have been muted. Corrected in `ui/theme.py`: `accent` -> vivid
   `#2E86FF`, `pos` -> vivid `#00C853`, `neg` -> vivid `#FF3B30`, `warn` ->
   vivid `#FFB300` (IBKR-strength saturation). Structure (radius, padding,
   font) stays soft/institutional as before -- only the semantic colors
   needed to punch through clearly, since that is the entire point of
   color-coding a data terminal.
2. **Correlation/crash-overlap heatmaps used an arbitrary amber-navy-blue
   scale.** Replaced with a new shared `MATRIX_COLORSCALE` constant
   (`ui/theme.py`) -- the conventional diverging red<->blue scale, using the
   same hues as `THEME["neg"]`/`THEME["accent"]` for consistency. Applied to
   both the Tab 1 dendrogram heatmap and the Tab 3 Jaccard/K heatmaps.
3. **Layout still "floated"** -- Section 5d only changed style *values*
   (radius numbers, hex codes), not the container *structure*: each stage
   panel remained its own independently-margined, independently-cornered
   box. Fixed two ways: (a) a mechanical pass across `ui/layout.py` reducing
   remaining large radii (up to 24px) down to 4px and tightening
   inter-section margins (35px -> 16px) throughout Tabs 1-3's sequential
   workflow panels; (b) a new `ui/components.py` function,
   `build_kpi_strip()`, which renders a list of KPIs as ONE bordered
   instrument strip with thin internal vertical dividers -- replacing
   `build_kpi_card()`'s per-item floating box wherever KPIs are shown in a
   row (Tab 4's results strip, Tab 5's portfolio-return strip and
   sandbox-metrics strip). `build_kpi_card()` itself is kept for any
   standalone/non-strip use. The wrapping Output divs in
   `ui/layout.py` (`stage4b-kpi-row`, `kpi-summary-row`) had their own
   grid/gap styling removed, since `build_kpi_strip()` is now a single
   self-contained component, not a list expecting a grid wrapper.

Verified: engine-level color value checks, a monkeypatched
`dash.callback_context` test of `build_kpi_strip()` rendering through the
real `render_stage4b_results` callback (confirms identical solver weights
to pre-change runs -- the redesign touched zero business logic), and a full
app integration recheck.

## 5f. Etap 3 — Sidebar Navigation (shell complete; module content pending)

Replaced the flat 5-tab bar's implicit "which module am I in" with an
explicit narrow sidebar (`ui/layout.py`), per the v2 spec's 4-module
architecture (Overview / Research / Portfolio Rebalance / Sandbox) --
matching the reference IBKR desktop screenshot's narrow icon column rather
than a wide labeled nav. Deliberately NO emoji or decorative icon glyphs:
each nav item is a plain bordered abbreviation box (`OV`/`RS`/`RB`/`SB`) +
a small text label underneath, as an explicit placeholder "until exact
icons are provided" (per instruction) -- swapping these for real icons
later only touches `_sidebar_item_children()`, nothing else.

**Structural change:** `app.layout`'s outer container became a flex row:
`SIDEBAR` (70px fixed) + a content wrapper carrying the page's original
padding/background. The former single `dcc.Tabs` (5 tabs) was split: Tabs
1-4 stay together inside `dcc.Tabs` (now wrapped in `html.Div(id="module-
rebalance")`, since that internal 4-step workflow -- Market Data ->
Fundamentals -> Tail-Risk -> Strategy -- genuinely is one sequential
process, matching the spec's "Portfolio Rebalance" module). Tab 5's content
was extracted from `dcc.Tabs` entirely (Dash's Tabs children must all be
`dcc.Tab` siblings sharing one `value`, so it could not stay inside while
becoming an independent top-level module) into its own
`html.Div(id="module-sandbox")`, matching the spec's standalone "Sandbox"
module. `module-overview` and `module-research` are simple "under
construction" placeholders -- their real content is out of scope for this
pass (planned Etap 4/5 per the original roadmap) but the navigation slot
and show/hide plumbing already exist, so building them out later is additive.

New `switch_active_module` callback (`ui/layout.py`) toggles which module
`Div` is visible (`display: block/none`) and which nav item is highlighted,
using `dash.callback_context` to determine which of the 4 `navitem-*`
components was clicked -- the same pattern already used by
`toggle_inspector_sidebar` (`ui/tab1_market_data.py`), kept consistent
rather than introducing a different navigation mechanism. Explicit Outputs
(13 of them: 1 store + 4 module styles + 4 navitem styles + 4 navitem
children) rather than a `STAGE4A_PARAMS_CONFIG`-style loop -- reasonable at
4 items mixing two different property types per item.

Default module on load: **Rebalance**, not Overview -- Overview is a
placeholder with no real content yet, so defaulting to it would show an
empty screen; Rebalance is where all the actually-functional Stage 1-4 flow
already lives. Revisit this default once Overview is built out (Etap 4/5).

Verified: `switch_active_module` tested by monkeypatching
`dash.callback_context.triggered` (a real Dash request context isn't
available outside a running server) for all 4 nav items plus the
no-trigger edge case (e.g. page load) -- each correctly shows exactly one
module, hides the other three, and highlights only the clicked nav item.
Component-tree check confirms exactly 4 `dcc.Tab` children remain (not 5 --
Tab 5 successfully extracted) and all new ids
(`navitem-*`/`module-*`/`store-active-module`) are present in the actual
rendered layout. Full app integration recheck: 35/35 callbacks (34 prior +
this one), solver regression unchanged.

---

## 5g. Design Correction #2 — Chart Palette, Matrix Legibility, Sandbox Gating (complete)

Follow-up after real screenshots surfaced four more problems left by Section
5e/5f's first pass:

1. **Multi-ticker overlay chart colors indistinguishable.** `CHART_COLORS`
   had only 6 muted hues (tuned for filling cluster/dendrogram cells, where
   subtlety across a handful of categories is fine); with 8+ tickers
   overlaid on one line chart, colors both repeated (cycling past 6) AND
   several of the 6 were too close in hue to tell apart at a glance (two
   ambers, two blue-teals). Expanded to 10 genuinely distinct vivid hues;
   `CLUSTER_PALETTE` now reuses the first 6 of these rather than
   maintaining a second, independently-tuned palette that could drift.
2. **Matrix colors: wrong direction, no value labels, harsh saturation, no
   cell borders, no separation between side-by-side matrices.** Multiple
   fixes:
   - `generate_tws_matrix_styles()` (Tab 1 correlation table) blended
     negative correlations toward `THEME["warn"]` (amber) instead of red --
     fixed to `THEME["neg"]`, per the standard -1=red/0=neutral/+1=blue
     convention. Blend intensity capped at 55% of full saturation (`MAX_BLEND`)
     so cells never reach raw neon red/blue even at |val|=1.
   - Both Plotly `go.Heatmap` calls (Tab 1 clustered heatmap, Tab 3 J/K
     crash-overlap) previously showed NO per-cell numeric value at all --
     the single biggest legibility problem, not really a color issue. Added
     `text=..., texttemplate="%{text:.2f}"` to both, matching the
     Excel-conditional-formatting look the request referenced.
   - Added `xgap=2, ygap=2` to both heatmaps for visible cell borders/grid
     lines (previously cells ran together with no separation).
   - New `ui/theme.py` split: `MATRIX_COLORSCALE` (gentle diverging,
     endpoints blended ~45% toward the dark background rather than hitting
     full-strength `THEME["neg"]`/`THEME["accent"]` -- "gentle like Excel"
     on a dark theme means bringing colors closer to the dark background,
     not lighter/whiter, which would fight the theme) for genuine -1..+1
     correlation data, and a new `MATRIX_SEQUENTIAL_COLORSCALE` (neutral
     background -> gentle blue only, no red pole) for non-negative 0..max
     data like the J/K crash-overlap matrices, which have no meaningful
     "negative" direction. Tab 1's heatmap now picks diverging vs
     sequential based on its own `zmin` (< 0 -> diverging, else sequential)
     since that function already computes both cases depending on
     clustering method.
   - Added a visible bordered panel around each of the J and K heatmaps
     (`ui/layout.py`) so it's unambiguous which matrix is which when shown
     side by side -- previously only a content gap separated them, easy to
     misread which heatmap belongs to which title at a glance.
3. **Sandbox panel invisible unless Stage 3 was just confirmed.**
   `panel-stage4b-tracker-container` was gated by a `reveal_stage4b_tracker_panel`
   callback (`Input("store-stage3-final-payload", "data")`) inherited from
   when Sandbox was tab-5 sharing sequential navigation with Tabs 1-4. Since
   Etap 3 made Sandbox its own independent sidebar module -- explicitly
   meant to load any saved snapshot from disk without touching the
   Rebalance workflow first -- that gate now hid the panel for exactly the
   use case it exists for. Callback removed entirely (not commented out --
   a dead Output/Input wiring left behind is worse than no trace); the
   panel is now always visible by default in `ui/layout.py`.

Verified: engine-level palette/color-value checks (10 distinct
`CHART_COLORS`, negative-correlation blend confirmed red-toned via RGB
component comparison), a real `render_crash_overlap_visuals` call
confirming `texttemplate`/`xgap`/`ygap` are actually set on the returned
Plotly figure objects (not just present in source), a layout tree check
confirming the Sandbox panel's default style no longer contains
`display: none`, a full app integration recheck (34/34 callbacks -- 35
minus the one intentionally removed), and a solver regression rerun
(identical weights to every prior check in this project's history).

---

## 5h. Etap 4 — Research / Company Dossier Module (complete)

Wires the Etap 1 data layer (`data/universe_store.py`, `data/company_store.py`
-- built with no UI consumer at the time) into Module 2's sidebar slot,
replacing its placeholder. Three-column IBKR-style layout, per the original
v2 spec and the concept mockup agreed with the project owner
(2026-09-07 conversation): search + company list (left) / dossier metrics +
target-vs-price history chart (center) / fundamental-data update form (right).

**Decisions confirmed before implementation:**
- **No valuation-ratio snapshot (P/E, P/S).** Investigated: yfinance has no
  historical time-series endpoint for these, only a live "today" `.info`
  snapshot; a single point-in-time number with no history was judged not
  worth adding. A forward-P/E computed from the manually-entered EPS 2Y
  CAGR was floated (`trailing_EPS * (1+CAGR)^0.5`) but dropped for the same
  reason -- explicitly out of scope for this pass.
- **New tickers auto-join the universe as "Watchlist"** the first time a
  snapshot is saved for them -- no separate "add to universe" step.
  (`ui/module_research.py`'s `save_snapshot` calls
  `universe_store.upsert_company()` unconditionally before
  `company_store.append_entry()`.)
- **Freshness threshold: 30 days fresh / 60 days stale**, matching the
  original spec's "<30 dni" wording, with an intermediate amber band (30-60
  days) rather than a hard binary so a company isn't flagged identically at
  31 days old vs 300.
- **Deliberately NOT wired into Stage 3's fundamental-inputs table.**
  Having the Rebalance workflow auto-pull from `company_store` instead of
  requiring manual re-entry is a real, valuable follow-up (matches the
  original spec's "Portfolio Rebalance zasila się automatycznie... dla
  spółek ze statusem Active_Screened") but is a distinct, larger change to
  an existing working data flow -- explicitly deferred, not bundled here.

**New file `ui/module_research.py`:**
- `render_company_list`: reads `universe_store.list_companies()`, filters
  by the search box, shows each as a clickable row with a freshness dot
  (`_freshness_dot_color`) driven by `company_store.days_since_last_update()`.
- `select_ticker`: a pattern-matching callback (`Input({"type":
  "research-company-row", "ticker": ALL}, "n_clicks")`) determining which
  row was clicked via `dash.callback_context`, OR the "+ ŚLEDŹ" button for
  typing a brand-new ticker not yet in any list. Guards against Dash's
  pattern-matching quirk where a newly-rendered row (e.g. after a search
  filter changes the matched set) can appear as a "phantom" trigger with
  `n_clicks=0`/`None` even though nothing was actually clicked -- checks the
  triggered value is truthy before treating it as a real click.
- `populate_dossier`: on ticker selection, reads `universe_store.get_company()`
  + `company_store.get_latest()`/`get_history()`, fetches a live P0 via
  `data.market_data.fetch_current_prices()` (single ticker), and fills the
  metrics grid, the target-vs-price history chart (empty with a placeholder
  annotation for a ticker with no saved history yet -- expected, not a bug,
  since this data only starts accumulating from when it's first saved), and
  every form field. A brand-new ticker (no `company_store` entry at all)
  gets blank form fields rather than erroring.
- `save_snapshot`: validates all 7 required fields are present (rejects the
  save with a specific "uzupełnij pola: ..." message naming which ones are
  missing, rather than silently writing partial/zero data), then calls
  `universe_store.upsert_company()` + `company_store.append_entry()`, and
  bumps `store-research-refresh` so the left-hand list re-renders with the
  new/updated freshness dot immediately.

`ui/layout.py`'s `module-research` placeholder replaced with the real
3-column structure (component ids: `research-search-input`,
`research-company-list`, `research-new-ticker-input`/`-btn`,
`store-research-selected-ticker`, `store-research-refresh`,
`research-dossier-header`/`-metrics`/`-chart`, `research-input-{name,
sector,status,p0,target,thigh,tlow,nanalysts,epscagr,epsrev}`,
`research-btn-save`, `research-save-status`). `app.py` gained one more
import (`ui.module_research`) alongside the 5 existing tab modules.

Verified: all new component ids present in the actual rendered layout
tree; `select_ticker` tested against 4 simulated `dash.callback_context`
scenarios (row click, new-ticker button, phantom zero-click trigger, empty
new-ticker input) via the same monkeypatch technique used for Etap 3's
`switch_active_module`; a full save -> read-back cycle against the real
`data/` layer (auto-adds to universe, `company_store` entry matches what
was submitted, list re-renders with the new company, validation correctly
rejects a save missing a required field); `populate_dossier` tested both
for an existing ticker with history and a brand-new one without; full app
integration recheck (38/38 callbacks -- 34 prior + 4 new); solver
regression rerun (identical weights to every prior check in this project's
history, confirmed to the same floating-point precision).

**Follow-up fix (same day):** two mechanic changes requested after initial review:
1. **"+ ŚLEDŹ" now adds the ticker to `universe_store` immediately**, before
   any fundamental data is entered -- previously a new ticker only reached
   the universe at the point `save_snapshot` was called, meaning it
   wouldn't appear on the left-hand list at all until the full 7-field form
   was filled in and saved. Tracking a company and entering its
   fundamental data are now two independent steps, as requested: the
   ticker shows up on the list right away (with a red "brak danych"
   freshness dot until a snapshot is actually saved for it).
2. **Name and Sector are now auto-fetched from yfinance** (`Ticker(ticker).info`,
   new `data/market_data.py` function `fetch_company_profile`) at the
   moment a new ticker is added via "+ ŚLEDŹ", rather than requiring manual
   entry. Degrades to empty strings (never raises) if yfinance has no
   profile data for a given ticker -- expected for some of the project's
   non-US tickers (KRX/ASX/LSE), where profile coverage is less reliable
   than daily price history; the fields stay user-editable either way.

Both `store-research-refresh` writers (`select_ticker`'s "+ ŚLEDŹ" branch
and `save_snapshot`) now declare `allow_duplicate=True`, required by Dash
whenever more than one callback targets the same Output.

A bug was introduced and caught during this fix: an early edit to insert
`fetch_company_profile` into `data/market_data.py` accidentally deleted the
`def fetch_current_prices(...):` signature line immediately below it,
leaving that function's docstring and body orphaned with no `def` --
caught immediately by the very next `python3 -m py_compile` /
full-app-import check (raised `ImportError: cannot import name
'fetch_current_prices'`), not by a later manual review. Restored and
reverified. Included here as the same category of lesson as Etap 3's
f-string/CSS-brace bug: mechanical text edits near existing code need a
real import/compile check immediately after, not just a visual diff read.

Verified: full app integration recheck (38/38 callbacks, unchanged count --
this was a mechanic fix, not a new callback), and a live test confirming a
brand-new ticker appears in `universe_store` (with fetched name/sector,
"Watchlist" status) immediately after "+ ŚLEDŹ" is clicked, while
`company_store` correctly has no entry for it yet until a snapshot is
separately saved.

---

## 5i. Etap 4 Follow-up — Single-Asset Projection & Diagnostic Engine (complete)

Extends the Research module (2026-09-07, second follow-up) into a full
per-ticker forward projection tool with a probability-cone chart and
historical backfilling, confirmed in detail with the project owner before
implementation.

**New `engine/single_asset.py`** -- pure math, mirrors `engine/returns.py`'s
Alpha Blend design (gamma coupled to the target-price branch only, drops
out at alpha=1.0; A_N discounts the whole bracket regardless of alpha) but
re-parameterized per-horizon (1M/3M/6M/1Y, each with its own default
alpha/gamma/kappa, shared N_ref=15) and extended with **eta** -- confirmed
design: a manual "Execution / Realization Factor", a multiplicative
correction for a company's historical track record of delivering vs.
missing guidance (0.80 for a chronic under-deliverer, 1.15 for a
beat-and-raise name), neutral default 1.0, flagged for a future automatic
realized-vs-expected comparison once enough history accumulates.
`compute_single_asset_projection()` returns the full forward cone
(P_proj/P_upper/P_lower via sigma_ann for the P90 side and delta_down_ann
for the P10 side -- the same upside/downside risk asymmetry philosophy as
the portfolio-level engine, just for one asset with no N x N matrix
machinery needed) plus diagnostics (P_profit via `norm.cdf`, Asymmetry_Ratio).

**`data/company_store.py` schema change, with full backward compatibility.**
Storage format changed from a bare JSON list to `{"entries": [...],
"horizon_profiles": {...}}`, to hold saved per-horizon {alpha, gamma,
kappa, eta} tuning overrides per ticker. A file written before this change
(bare list) is read transparently as `{"entries": <that list>,
"horizon_profiles": {}}` -- no migration script, no one-time conversion;
the next write naturally upgrades it. `get_horizon_profile()` returns
`None` (not an engine-level default) when nothing is saved -- deliberately
NOT importing `engine/single_asset.py` to supply that default itself,
keeping the one-directional dependency rule (`data/` never imports
`engine/`) intact; `ui/module_research.py`'s `load_horizon_sliders` is what
bridges the two, falling back to `DEFAULT_HORIZON_PARAMS[horizon]` when the
saved profile is `None`.

**Historical backfilling.** `research-input-date` (new `dcc.DatePickerSingle`,
max date = today) lets a snapshot be saved under ANY past date, not just
today -- `append_entry()` already supported an explicit `entry_date` from
Etap 1, this exposes it in the UI. P0 now follows the selected date: today
(or later) triggers a live yfinance fetch; a past date looks up the
nearest close AT OR BEFORE that date from the already-cached 5Y series
(`_lookup_price_asof`) -- no extra network call, reuses the same cache the
chart's background line uses. `entries` stays chronologically sorted by
Date on every write (already true since Etap 1; verified again here) so a
backfilled point lands in the correct position for the chart regardless of
entry order, and re-saving an already-used date upserts in place (status
message distinguishes "Zapisano nowy wpis" vs "Zaktualizowano istniejący
wpis" so the user knows which happened).

**UI additions (`ui/layout.py` + `ui/module_research.py`):** a horizon
`dcc.RadioItems` (1M/3M/6M/1Y) above the chart; the chart itself now draws
the 5Y price line, historical Ti/Thigh/Tlow points from saved snapshots,
AND the live forward cone (`fill="tonexty"` between P10/P90, matching the
approved diverging-blue accent fill) with horizontal reference lines for
the current Target Consensus/High/Low; a `build_kpi_strip()` row below the
chart (Standalone Return μ(h), Win Probability P(R>0), Asymmetry Ratio,
Downside Volatility) reusing the same connected-instrument-strip component
from Etap 3's design correction, not a new one-off; a collapsible "Model
Tuning" panel (4 sliders + a "Zapisz profil dla tego horyzontu" button)
that loads/saves through `company_store.get_horizon_profile`/
`save_horizon_profile`.

**Callback graph note:** `store-research-price-history` (a new `dcc.Store`
caching the 5Y series once per ticker selection, reused by the chart's
background line, the engine's sigma/delta_down estimation, AND the
backfill price lookup) is populated by its own `cache_price_history`
callback, decoupled from `populate_dossier` and `sync_p0_with_date` --
Dash sequences the graph correctly because those two depend on the cache
(as a `State`) or on `research-input-date` changing (which
`populate_dossier` itself triggers by resetting the date to today on ticker
selection), not because of manual ordering.

Verified: engine-level numeric tests (gamma vanishes at alpha=1.0 exactly
as in the portfolio-level formula, all 4 horizons resolve their own
defaults, partial parameter overrides preserve the rest of a horizon's
defaults, eta scales mu_h linearly, divide-by-zero guards for P0=0/short
history, invalid horizon raises `ValueError`); `company_store` backward
compatibility (an old bare-list file reads correctly, auto-upgrades to the
wrapped format on next write, horizon-profile CRUD, chronological sort
after backfilling out of order, upsert-not-duplicate on a repeated date);
full projection-chart-and-KPI rendering with real trace/KPI-count checks
and a no-crash check when required fields are missing; the complete
backfill-save flow end to end (new past-dated entry, re-save of the same
date correctly reports "updated" not "new", final history chronologically
sorted). Full app integration recheck (44/44 callbacks -- 38 prior + 6
net-new here) and a solver regression rerun (identical weights to every
prior check in this project's history).

---

## 5j. Walk-Forward Calibration & Backfit Engine (complete)

Extends the Research module (2026-09-08, third follow-up) from a static
"projection in a vacuum" calculator into a genuine backtest/calibration
tool: uses the fundamental snapshots already saved for a ticker
(`data/company_store.py`'s history) to check how well the model's past
predictions matched what actually happened once each horizon elapsed, and
lets those predictions be automatically re-fit against that history.

**`engine/single_asset.py` refactor + additions.** `_compute_mu_h()` was
extracted as its own lean function (formula only, no cone/diagnostics) --
`compute_single_asset_projection()` now just calls it once for display;
the new backtest/fit functions call it many times (once per historical
entry, and for fitting, once per entry per optimizer iteration) without
paying for cone construction on every call. Verified byte-for-byte
unchanged output from `compute_single_asset_projection()` before vs after
this refactor (same mu_h/P_profit/Asymmetry_Ratio on a fixed test case).

- `evaluate_historical_accuracy(history_entries, price_series, horizon, params)`
  -- for each saved snapshot where `horizon` TRADING SESSIONS (not calendar
  days) have already elapsed, compares the model's predicted return against
  what actually happened. Session-based advancement
  (`_target_date_after_sessions`) walks the asset's own real trading
  calendar (its 5Y price series' dates) rather than naively adding calendar
  days, which would systematically misalign around weekends/holidays.
  Entries where the horizon hasn't elapsed yet are silently skipped, not
  included with null placeholders.
- `fit_single_asset_parameters(history_entries, price_series, horizon)` --
  `scipy.optimize.minimize` (L-BFGS-B, box-bounded per confirmed
  `FIT_BOUNDS`: alpha∈[0,1], gamma∈[0,1.5], kappa∈[0,2.0], eta∈[0.5,2.0]),
  minimizing mean squared return-prediction error across every evaluable
  entry. `N_ref` stays fixed (not part of theta, confirmed). Returns `None`
  below `MIN_ENTRIES_FOR_FIT=2` evaluable entries -- fitting 4 free
  parameters to 0-1 points is meaningless, not just imprecise.

**A real, important finding surfaced during testing (not a bug): parameter
non-identifiability at small sample sizes.** A zero-noise synthetic test
(generate 8 historical entries + resulting "true" prices from a KNOWN
theta, then fit) initially failed to recover the true theta (gamma off by
0.07, kappa off by 0.5) despite the optimizer finding a near-perfect
MSE≈7.6e-7 minimum. Investigation confirmed this is genuine model behavior,
not an optimizer bug: several different (alpha, gamma, kappa, eta)
combinations can produce nearly identical mu_h for a small/homogeneous set
of inputs (the same test re-run with 30 more diverse synthetic entries
recovered the true theta to within 0.0004 on every parameter). Practical
consequence, surfaced directly in the UI: `fit_single_asset_parameters`
returns `n_evaluable` and `mse` specifically so a small-sample fit can be
flagged rather than presented with false confidence --
`run_autofit`'s status message adds an explicit caveat whenever
`n_evaluable <= MIN_ENTRIES_FOR_FIT + 1`.

**UI: two charts + a backtest table, per the confirmed design.**
- **Chart A** (`research-dossier-chart`, `render_main_chart_and_kpis`): 5Y
  price line, saved historical Target Consensus/High/Low points, historical
  model predictions plotted at their OWN evaluation date (`evaluate_historical_accuracy`'s
  `p_pred`/`date_target` -- "what did the model think a year ago about
  today, vs. what happened"), and the live forward cone. **Confirmed
  correction:** the cone's P0 anchor is now ALWAYS the most recent price in
  the cached 5Y series (`price_series.iloc[-1]`), decoupled from
  `research-input-p0` (which follows the backfill DatePicker and could hold
  a stale historical value) -- verified directly: the projection's t=0
  point matches the cache's last price exactly, not whatever the form
  currently shows. `research-input-p0` was removed from this callback's
  Inputs entirely (no longer relevant to what it renders).
- **Chart B** (`research-backtest-chart`, `render_backtest_chart_and_table`):
  dual-axis bar+line -- green/red bars for per-entry return error (colored
  by `error > 0` = model overshot = red, matching the same sign convention
  used in the table), a Realization Ratio line on a secondary y-axis. Depends
  only on already-saved history + the current horizon/sliders, NOT on the
  "new entry" form fields (a backtest looks backward at saved data, not at
  data not yet saved).
- **Backtest Inspection Table**: the 9 columns as specified, using the same
  shared `datatable_style_*` helpers as every other table in the app for
  visual consistency. **A real bug found and fixed during testing:** the
  Status column's "Dowiezione/Niedowiezione: X%" label is mathematically
  correct but reads as nonsense when the model and reality point in
  OPPOSITE directions (e.g. model predicted +11%, reality was -4%, giving a
  literal "-37%" realization) -- added a distinct "Chybione (przeciwny
  kierunek)" label for exactly that sign-flip case, verified against both a
  same-direction case (normal percentage label) and an opposite-direction
  case (new label) using hand-constructed scenarios.
- **`research-btn-autofit`** ("⚡ AUTODOPASUJ PARAMETRY POD HISTORIĘ"):
  writes the fitted values directly to all 4 sliders
  (`allow_duplicate=True`, since the sliders are also written by
  `load_horizon_sliders` when switching ticker/horizon), which in turn
  retriggers both charts and the table automatically via the existing
  slider-Input wiring -- no separate "recompute" step needed.
- Slider ranges widened to match `FIT_BOUNDS` exactly (gamma 0-1→0-1.5, eta
  0.5-1.5→0.5-2.0) -- a real gap caught before it caused a bug: a fitted
  value outside the old slider range would have been silently clipped by
  the `dcc.Slider`, misrepresenting the actual fit result.

Verified: engine-level tests (trading-session advancement vs. naive
calendar-day math, the zero-noise parameter-recovery sanity check at both 8
and 30 synthetic entries, the `MIN_ENTRIES_FOR_FIT` guard); full
chart/table rendering against realistic synthetic history (backtest table
row count matches evaluable-entry count, Chart A carries both the
shifted-historical-prediction trace and the live-anchored cone trace, Chart
B has exactly 2 traces); the live-price-anchor fix confirmed numerically
(cone's t=0 value equals the cache's last price to the cent); the Status
label fix re-verified after the change; full app integration recheck
(46/46 callbacks -- 44 prior, net +2 here); solver regression rerun
(identical weights to every prior check in this project's history).

---

## 5k. Parameter Stability Measure (complete) — methodology change from the previous section

The project owner asked directly: how many historical observations are
needed for `fit_single_asset_parameters` to mean anything, and can
stability be measured rather than guessed? This triggered a genuine
methodology correction to Section 5j's fitting approach, not just an
additive feature.

**First attempt: bootstrap resampling — implemented, tested, and abandoned
with the reasoning kept in `engine/single_asset.py`'s docstrings.** Wrapped
the existing fit in `n_bootstrap=200` resamples-with-replacement, reporting
the empirical std/CV of each parameter across replicates. Tested against
the exact two zero-noise synthetic scenarios from Section 5j (8 points:
known non-identifiable; 30 diverse points: known well-identified) --
**bootstrap reported "Stabilne" for every parameter in the 8-point case**,
completely failing to flag the known-bad fit. Root cause: resampling with
replacement from a small, FIXED set of points only reweights the SAME
underlying feature combinations -- it cannot reveal a degeneracy that is a
property of insufficient feature diversity itself (as opposed to sampling
noise), because every resample still only ever contains combinations drawn
from that same fixed pool.

**Root cause investigation surfaced a second, more important bug:** cross-checking
with `scipy.optimize.least_squares` (Trust Region Reflective) on the SAME
8-point case that `fit_single_asset_parameters`'s original
`scipy.optimize.minimize(..., method="L-BFGS-B")` had gotten wrong found
the EXACT true parameters with zero error. This means the earlier "wrong"
fit was not (only) a fundamental model non-identifiability -- it was an
**optimizer robustness failure**: generic scalar-objective L-BFGS-B from a
single starting point got stuck away from the true optimum on this
problem's loss surface, while `least_squares` (purpose-built for
sum-of-squared-residuals problems, which this literally is) found it
reliably.

**Fix: `fit_single_asset_parameters` rewritten around `scipy.optimize.least_squares`**
instead of `minimize` on a hand-rolled scalar MSE. This is both the
mathematically correct tool for this exact problem class AND yields the
Jacobian at the solution for free, enabling the classical nonlinear-least-squares
asymptotic covariance estimate: `Cov(theta) ≈ sigma² · (JᵀJ)⁻¹`, `sigma² =
SSE/(n_obs - 4)`. Per-parameter standard error and coefficient of variation
(`cv = stderr / |estimate|`) are now returned alongside the point estimate,
plus the condition number of `JᵀJ` (a high condition number flags a
near-flat/degenerate fitting direction even before inspecting any single
parameter). Re-verified against the same two synthetic scenarios plus a
NEW, more realistic one with 3% Gaussian noise added to the simulated
outcomes (the original zero-noise tests turned out to be a poor testbed for
validating a stability measure in their own right -- at exactly zero
residual, the asymptotic covariance formula's `sigma²` term collapses
toward zero regardless of the true identifiability of the problem, making
any method look artificially confident). Under realistic noise: 8 points
correctly showed enormous stderr (e.g. kappa: 0.000 ± 6.121, completely
unconstrained within its bounds); 30 diverse points showed visibly tighter
(alpha/eta became "Stabilne") but still appropriately wide intervals for
harder-to-pin-down parameters (gamma/kappa) -- exactly the graduated,
honest signal a stability measure should give, and the fix's zero-noise
recovery test now passes exactly (all four parameters recovered to
<0.00001 on the original problem case).

**UI:** each of the 4 tuning sliders (`ui/layout.py`) gained a small
stability badge below it (`research-stability-{alpha,gamma,kappa,eta}`),
populated by `run_autofit` after each fit with a plain-language tier
(`engine.single_asset.stability_label`: "Stabilne" / "Umiarkowanie
stabilne" / "Niestabilne" / "Bardzo niestabilne / niezidentyfikowane",
thresholded on CV) plus the CV itself. **A real display bug found and
fixed during testing:** an extremely under-determined fit can produce
astronomical CV values (observed: `3.4e16`%) that are mathematically
correct but read as a glitch, not a diagnostic -- capped the displayed
figure at `"CV>1000%"` once `cv > 10` rather than printing the literal
number; the qualitative label itself is unaffected. Badges are cleared
(not left stale) by `load_horizon_sliders` whenever the ticker or horizon
changes, since a stability verdict from a previous fit context doesn't
apply to a new one.

Verified: the corrected zero-noise recovery test (exact parameter recovery
on the original 8-point case that had previously failed), the new
realistic-noise stability comparison (8 vs. 30 points, both the point
estimates and the stderr/CV behaving in the expected direction), the
capped-CV display fix (spot-checked against both an astronomically
unstable case and a moderately-diverse 8-point case with plausible,
non-absurd CV values), `load_horizon_sliders`'s badge-clearing behavior,
full app integration recheck (46/46 callbacks -- same count as Section 5j,
since this was a rewrite of existing callbacks' outputs rather than new
callback registrations), and a solver regression rerun (identical weights
to every prior check in this project's history).

---

## 5l. Cointegration Pair Screener (`engine/pairs.py`, complete — screening only, not yet wired to mu_i)

Standalone Engle-Granger pair screener, rewritten from scratch after
auditing a third-party (Gemini-authored) reference script the project
owner had been experimenting with -- three confirmed bugs found by
inspection, all avoided here:

1. **The reference script's stated "p-value < 0.05" gate was never actually
   applied.** Its filtering loop only checked `if 5 <= half_life <= 90:`
   before appending a pair to the ranked table -- `adf_p` was computed and
   displayed but never compared against any threshold. This directly
   explained a discrepancy the project owner flagged earlier: MU vs WDC
   (p=0.0958, nearly 2x the stated threshold) appearing in the "top 20"
   table despite failing the very criterion the accompanying report
   described.
2. **No sanity check on the hedge ratio's sign or magnitude.** A negative
   hedge ratio (observed in the reference script's own output: GRAB vs
   MCHP, gamma=-0.608) means the two series move in OPPOSITE directions --
   not economically a "pair" for a same-direction relative-value mechanism,
   almost certainly a spurious regression that happened to pass every other
   filter.
3. **Plain `statsmodels.tsa.stattools.adfuller()` applied directly to OLS
   regression residuals**, using ADF's standard critical values -- which
   assume a raw observed series, not residuals from an estimated
   2-parameter regression (which have a different, non-standard null
   distribution). This systematically overstates evidence for
   cointegration. Fixed by using `statsmodels.tsa.stattools.coint()`
   instead, which applies the correct MacKinnon-adjusted critical values
   for exactly this two-step Engle-Granger setup.

**Confirmed project philosophy (important, shapes every design choice
here):** this is explicitly NOT classical market-neutral pairs trading. The
project is Long-Only, rebalances on the existing ~21-trading-day (monthly)
cadence -- a qualifying pair's Z-score will eventually give a bounded,
SOFT nudge to both names' `mu_i` (via `theta * tanh(-Z)`, agreed in the
immediately preceding conversation turn as the safe replacement for a
proposed-but-flawed unbounded linear `mu*(1-theta*Z)` correction that could
flip mu's sign at extreme Z or push a genuinely negative mu further
negative on a "cheap" reading) -- evaluated once per rebalance cycle, not
continuously, and never with hard 80/20 weight-switching execution. A
"wrong" pair costs at most one weaker month on one hyper-growth name, not a
leveraged market-neutral blowup. This screener produces the CANDIDATE PAIR
LIST only; the mu_i adjustment itself is a separate, not-yet-implemented
step.

**Four hard gates, in cheapest-first order** (`find_cointegrated_pairs`):
1. 2Y drift gate: `|R_A(2Y) - R_B(2Y)| <= 35pp` (run before the expensive
   cointegration test, so a pair that was always going to fail on drift
   alone never reaches it).
2. Engle-Granger cointegration: `p_value < 0.05` via `statsmodels.coint()`.
3. Half-life bound: `5 <= half_life <= 63` trading sessions (from AR(1) fit
   on the OLS residual spread) -- kept exactly as stated in the project
   owner's original report for now; revisiting this range for the
   monthly-cadence use case (as opposed to a report written with faster
   arbitrage-style trading in mind) is an explicit, separate decision for
   later, not silently changed here.
4. Hedge ratio sanity: `0.3 <= gamma <= 3.0` (rejects near-zero, negative,
   or wildly-scaled hedge ratios -- a spurious-regression signature).

Every gate is a hard reject; a pair failing any one never reaches the
returned table.

Verified with synthetic data (four constructed scenarios, using
statistically independent random-walk components for each -- an earlier
draft of the test accidentally reused the same underlying trend series
across two scenarios, producing a misleading cross-pair "false positive"
that was a test-construction bug, not a screener bug; rebuilt with
genuinely independent components before drawing conclusions):
- A genuinely cointegrated pair (shared stochastic trend + AR(1)
  mean-reverting spread) passes every gate.
- An independent-random-walk (spurious) pair is correctly rejected at the
  p-value gate (p=0.596).
- An excessive-2Y-drift pair is correctly rejected before the cointegration
  test even runs.
- **The cleanest single proof of gate 4's necessity:** a pair constructed
  to have a genuinely negative hedge ratio was shown to PASS the
  cointegration test outright (p<0.05 -- a real statistical relationship
  exists) and is rejected ONLY by the hedge-ratio sign gate -- direct
  evidence this check catches something the p-value test alone would let
  through, not a redundant/cosmetic filter.
- A full `find_cointegrated_pairs` run across all four synthetic pairs
  together, with `warnings.simplefilter("error")`, produces zero runtime
  warnings and returns exactly the one genuinely-qualifying pair -- confirms
  the four gates compose correctly as a pipeline, not just individually.
- A near-zero AR(1) coefficient (rho) in the half-life calculation was
  found, during testing, to trigger a numpy divide-by-zero `RuntimeWarning`
  even though the existing `rho >= 0` guard should have caught it (the
  warning fired for rho asymptotically close to but not exactly 0);
  tightened the guard to `abs(np.log1p(rho)) < 1e-10` so the check is
  robust to floating-point near-misses, not just exact zero.

New dependency: `statsmodels>=0.14` added to `requirements.txt` --
installed and confirmed importable in this environment; not previously a
project dependency.

**Not yet done, explicitly deferred:** wiring a qualifying pair's live
Z-score into an actual `mu_i` adjustment (the `theta * tanh(-Z)` mechanism
agreed in conversation, not yet implemented in code), and any UI/Dash
surface for this screener (no `ui/` changes in this pass -- this is
engine-layer only, run and verified standalone). Both are natural next
steps once the project owner has reviewed real screener output against the
project's actual ~40-ticker universe.

---

## 6. Etap 5 — Universum napędza Tab 1 (complete)

**Numbering note:** sections 5d–5l above accumulated a long, ever-growing
chain of sub-lettered follow-ups under "Etap 5" (design corrections,
sidebar, Research module, walk-forward engine, stability measure, pairs
screener) — the project owner flagged this as unsustainable (2026-09-08).
Going forward, each new major undertaking gets its own fresh top-level
Etap number instead of another sub-letter. This is Etap 5 (renumbered from
an earlier draft plan that called it "5a"); Etap 6 and Etap 7 are its two
planned siblings from the same conversation (a new "Relative Value"
cointegration module, and save-buttons in Rebalance/Research), not yet
implemented as of this section.

First of three planned, independent pieces (confirmed scope, 2026-09-08):
finishes wiring `data/universe_store.py` (built in Etap 1, previously only
consumed by the Research module) into the Rebalance workflow's own ticker
selection, replacing Stage 1's free-text `dcc.Textarea` with a searchable,
checkbox-based universe picker — **unchecked by default** (confirmed
explicitly: the project owner picks ~30-40 of an eventual ~160-name
universe per rebalance, so defaulting to all-checked would mean unchecking
~120 every session instead of checking the ~30-40 actually wanted).

**`ui/layout.py`:** `input-tickers-raw` (the old textarea) replaced with: a
search box (`stage1-universe-search`) filtering a `dcc.Checklist`
(`checklist-universe-tickers`) populated from `universe_store.list_companies()`;
a live "X zaznaczonych / Y w uniwersum" counter; and a mini "+ DODAJ DO
UNIWERSUM" form (ticker input + button) mirroring the Research module's
"+ ŚLEDŹ" pattern exactly — same auto-fetch-Name/Sector-via-yfinance
mechanism (`data.market_data.fetch_company_profile`), same
`universe_store.upsert_company(..., status="Watchlist")` call. No `Status`
filtering is applied to what's shown in this checklist (every tracked
company appears regardless of Active_Screened/Watchlist) — the project
owner was explicit that the Active_Screened/Watchlist distinction isn't a
concept they want surfaced here; the checkbox state itself (fresh per
session, nothing persisted) is the only "am I using this ticker in this
rebalance" signal for Tab 1.

**`ui/tab1_market_data.py`:**
- New `render_universe_checklist(search, refresh, checked_values)`: populates
  ONLY the Checklist's `options` from the (optionally search-filtered)
  universe — **never writes to `value`**, so filtering the search box can
  never clear a selection made before or after filtering (verified
  directly, see below — not just asserted).
- New `add_new_universe_ticker(...)`: adds a brand-new ticker to
  `universe_store` and immediately appends it to the currently-checked list
  (adding a ticker mid-session almost certainly means wanting it in THIS
  rebalance too, not just registered for later).
- `run_stage_01_ingestion` (the existing Stage 1 ingestion callback,
  otherwise completely unchanged) now reads `checklist-universe-tickers`'s
  `value` (already a clean list of ticker strings) instead of parsing a
  free-text textarea (`raw_input.replace(",", " ").split(" ")` — that
  parsing logic is gone entirely, made obsolete by the Checklist's own
  `value` shape). A real, if minor, UX regression was caught and fixed
  during this change: the old textarea shipped with a pre-filled default
  value, so submitting it empty was rare; the new checklist starts EMPTY by
  design, making "click INITIALIZE DATASETS with nothing checked" a much
  more likely first-time mistake. The old code's single combined check
  (`if n_clicks == 0 or not raw_input`) collapsed both "never clicked" and
  "clicked with nothing entered" into the same silent no-op with an empty
  error string; split into two checks so the now-more-likely "clicked with
  nothing selected" case gets its own message ("Zaznacz przynajmniej jedną
  spółkę z uniwersum.") instead of silently doing nothing.

Verified: all 6 new component ids present in the rendered layout tree;
`render_universe_checklist` confirmed to filter `options` correctly by
search AND — the specific behavior that actually matters here — confirmed
that during an active search filter, the "X zaznaczonych" counter still
correctly reports previously-checked tickers that are no longer visible in
the filtered options list (this is not just an assertion on the return
value: the callback's `Output` list *structurally* never includes
`checklist-universe-tickers.value`, so there is no code path by which it
could clear a selection — verified both by reading the Output list and by
exercising the function with a filtered search alongside a fixed checked
list); `add_new_universe_ticker` confirmed to add to `universe_store` AND
extend the checked list in one step (yfinance calls correctly degrade to
empty name/sector under this sandbox's network restrictions, as designed,
without crashing the flow); `run_stage_01_ingestion`'s three edge cases
(never clicked, clicked with `None`, clicked with `[]`) each verified to
produce the correct distinct outcome. Full app integration recheck (48/48
callbacks -- 46 prior + 2 new), solver regression rerun (identical weights
to every prior check in this project's history).

**Not yet done, explicitly deferred to Etap 6 and Etap 7** (separate,
independent pieces from the same conversation, per the confirmed plan):
- Etap 6: a new "Relative Value" module (renamed from an earlier
  "Pair Trading Research" working title — deliberately not "pair trading",
  since that phrasing implies the classical market-neutral strategy this
  project explicitly rejected) surfacing `engine/pairs.py`'s cointegration
  screener over the full universe, with its own independent price fetch
  (not reusing Tab 1's session-scoped `store-raw-close`, since the screener
  must run over the WHOLE universe regardless of which ~30-40 tickers are
  checked for any given rebalance).
- Etap 7: a portfolio-save button in Rebalance's Stage 4B results panel
  (using the existing `data/snapshot_store.py`, just exposed nearer the
  result rather than only in Sandbox), and two buttons in Stage 3's
  fundamental-inputs table ("Przejdź dalej bez zapisu" / "Zapisz i przejdź
  dalej" — the latter saving each table row individually to
  `company_history/{ticker}.json` via the existing
  `company_store.append_entry()`, dated today).

---

## 7. Etap 6 — Relative Value Module (screener wired to UI, diagnostic only)

Second of the three pieces confirmed in the 2026-09-08 conversation
("Etap 5/6/7" — see Section 6's numbering note). Surfaces
`engine/pairs.py`'s cointegration screener (Section 5l) as a real,
usable UI feature for the first time — that module had been engine-layer
only, verified with synthetic data, with no Dash surface at all until now.

**New sidebar module, "Relative Value"** (deliberately not "Pair Trading"
— the project owner's own naming choice, since that phrasing implies the
classical market-neutral long/short strategy this project's cointegration
report explicitly rejected). Fifth item in `SIDEBAR_ITEMS`
(`ui/layout.py`), inserted between Rebalance and Sandbox. Extending the
sidebar from 4 to 5 modules required touching `switch_active_module`'s
`@app.callback` decorator (which lists every module/navitem id explicitly,
by design — see Section 5f's note on why this isn't a
`STAGE4A_PARAMS_CONFIG`-style loop) and its function signature; the
function BODY itself needed no change, since it already iterates
`SIDEBAR_ITEMS` generically rather than hardcoding a count anywhere.
Re-verified all 5 nav positions individually after the change (not just
the new one) — each correctly shows exactly one module and hides the
other four.

**Scope, confirmed explicitly:** diagnostic only in this pass. The scanner
identifies candidate pairs and shows the same ranked table structure as
`engine/pairs.py` (Ticker A/B, 2Y drift, p-value, half-life, hedge ratio,
Score) — it does NOT yet feed into Rebalance's `mu_i` or Stage 1 SLSQP in
any way. The `theta * tanh(-Z)` mu-adjustment mechanism agreed earlier in
conversation is an explicit, separate, later step, deferred until the
project owner has reviewed real screener output against their actual
universe.

**Independent price fetch, by design (not a shortcut).** The scan
(`ui/module_relative_value.py`'s `run_relative_value_scan`) fetches prices
for EVERY ticker in `universe_store` via `data.market_data.fetch_universe_prices`,
completely independent of whatever subset is checked in Rebalance's Stage 1
`checklist-universe-tickers` (Section 6) for the current rebalance cycle.
This directly reflects the confirmed intended pipeline order: pair
screening happens across the WHOLE universe FIRST, and only what doesn't
sensibly pair up goes through the existing Ward/DTW/RMT clustering in
Rebalance — a pair candidate must be discoverable regardless of which
~30-40 names happen to be checked for any given month's rebalance.

**No background-callback infrastructure in this pass (confirmed scope).**
A full scan across a large universe (100+ names, ~C(160,2)=12,720 pairs at
the project owner's eventual target size) is a genuine multi-minute
blocking wait, not a UX bug -- `dcc.Loading` wraps the results `Output` so
the spinner shows for the duration. Revisiting this with real async/background
execution is a possible future improvement if the wait proves impractical
in practice, not addressed now.

Verified: `run_relative_value_scan` tested end-to-end against synthetic
data via a monkeypatched `fetch_universe_prices` (no network access in this
sandbox) — a 4-ticker synthetic universe (1 genuinely cointegrated pair + 2
independent/spurious tickers) correctly returns exactly the one qualifying
pair in the rendered `dash_table.DataTable`, with the independent tickers
correctly absent; the "fewer than 2 tickers in universe" guard tested and
confirmed; all 6 new component ids (`module-relval`, `navitem-relval`,
`btn-relval-scan`, `relval-scan-status`, `relval-results-table`,
`loading-relval-scan`) confirmed present in the rendered layout tree; all 5
sidebar positions (not just the new one) individually re-verified after
extending `switch_active_module`; Rebalance's own tab structure confirmed
unchanged (still exactly 4 `dcc.Tab` children); full app integration
recheck (49/49 callbacks — 48 prior + 1 new); solver regression rerun
(identical weights to every prior check in this project's history).

**Not yet done, still explicitly deferred to Etap 7** (per the confirmed
plan): a portfolio-save button in Rebalance's Stage 4B results panel, and
two buttons in Stage 3's fundamental-inputs table ("Przejdź dalej bez
zapisu" / "Zapisz i przejdź dalej", the latter saving each table row to
`company_history/{ticker}.json`).

### 7a. Etap 6 extension (same day) — near-miss visibility + interactive pair simulation

The initial Etap 6 scan only surfaced fully-qualifying pairs, using a
plain results table. The project owner asked for three concrete
additions, all implemented: (1) visibility into pairs that almost
qualified ("np. przeszły 3 bramki"), (2) an equity-curve visualization
matching the passive-50/50 vs 100%/0% vs dynamic-threshold-switching
comparison from the audited reference script, and (3) the ability to pick
ANY two tickers directly (not just ones from the ranking) and interact
with adjustable thresholds (e.g. 80/20 -> 85/15) to see how the pair behaves.

**`engine/pairs.py` refactor + two new functions:**
- `evaluate_pair_diagnostics(prices_a, prices_b, ...)`: full gate-by-gate
  diagnostics for ONE pair, deliberately NOT short-circuiting among gates
  2-4 (cointegration / half-life / hedge ratio) -- returns every gate's
  pass/fail boolean plus its underlying value, `gates_passed` (0-4), and
  `all_passed`. Gate 1 (drift) is still evaluated as a hard PRE-filter by
  callers scanning a whole universe (a pair with a wildly divergent 2Y
  trajectory is a fundamentally bad candidate regardless of its other
  statistics, and the expensive cointegration test shouldn't be spent on it).
- `scan_universe_diagnostics(prices_df, tickers, ...)`: full-universe
  near-miss scan -- every pair surviving the cheap drift pre-filter, with
  complete diagnostics, sorted by (gates passed desc, Score asc). More
  expensive than the original short-circuiting scan (every drift-surviving
  pair now pays for the full cointegration test even if it will fail
  later), an explicit, accepted tradeoff for near-miss visibility.
- `find_cointegrated_pairs` refactored into a thin filter
  (`gates_passed == 4`) over `scan_universe_diagnostics`'s output -- single
  source of truth for the gate logic now, rather than a second, separately
  maintained scan loop. **Verified by regression test to return byte-identical
  results to the original short-circuiting implementation** on the same
  4-scenario synthetic test suite used in Section 5l.
- A real test-construction mistake was caught while verifying near-miss
  behavior: a synthetic "negative hedge ratio" pair, expected to show "3 of
  4 gates passed" (failing only the hedge-ratio gate), instead showed 2 of
  4 -- investigation confirmed this was NOT a code bug: a pair with
  opposite-direction price co-movement (negative hedge ratio) will almost
  always ALSO show a large 2Y drift difference, since moving in opposite
  directions compounds into wildly different cumulative returns over two
  years. The gate-independence assumption in the original test design was
  wrong, not the screener; `evaluate_pair_diagnostics`'s output was checked
  directly against the sum of its own individual pass/fail flags and found
  exactly consistent (2 = sum([False, True, True, False])), confirming the
  counting mechanism itself is correct.

**New function, `simulate_pair_strategy(...)`:** a discrete-event,
threshold-switching Long-Only backtest for interactive exploration --
explicitly a DIAGNOSTIC tool distinct from (not a preview of) the eventual
monthly `theta*tanh(-Z)` mu-adjustment mechanism still deferred. State
machine matches the confirmed 80/20-style design (entry_z / exit_z /
favour_weight all adjustable). Z-score uses a ROLLING window on the OLS
spread (distinct from the STATIC full-sample regression used for the
screening p-value -- a live trading signal should adapt to recent spread
dynamics; the screening test needs a fixed window to be statistically
valid). Execution ported from the audited reference script's ALREADY-CORRECT
physical-share, discrete-event logic (this was not one of that script's
three bugs) -- verified independently rather than assumed correct:
- **No "Shannon's Demon"**: on two genuinely independent random walks (no
  real pair signal), the 50/50 benchmark's final value falls strictly
  inside the corridor bounded by the two components' individual Buy & Hold
  outcomes -- confirmed numerically, not just visually plausible.
- Tighter thresholds produce more state transitions (18 vs 93 transitions
  across two threshold settings on the same synthetic cointegrated pair).
- `favour_weight` (0.80 vs 0.85) produces genuinely different realized
  weight-history values, confirmed by direct inspection of the weight series.
- Higher `fee_bps` produces strictly lower final equity on a
  high-turnover (tight-threshold) run, confirmed numerically.

**A real display bug found and fixed during UI testing:** the chart's
strategy-curve legend showed "Dynamiczny Long-Only (80/19)" instead of
"(80/20)" at the default `favour_weight=0.80` -- caused by `int()`
truncating rather than rounding: `(1 - 0.80) * 100` evaluates to
`19.999999999999996` in IEEE754 floating point, and `int()` truncates that
toward zero, landing on 19. Fixed by using `round()` instead of `int()` for
this display formatting; re-verified both the 80/20 and 85/15 cases render
correctly. Caught by an assertion in the test suite, not by visual
inspection -- a reminder (same lesson as Etap 3's f-string bug and Etap 5h's
accidentally-deleted function signature) that floating-point/display
formatting deserves an explicit check, not just "looks right" on one sample value.

**UI (`ui/layout.py`, `ui/module_relative_value.py`):** the scan results
table now shows every drift-surviving pair with a "Bramki" (Gates) column
("4 / 4", "2 / 4", etc.) and a Status column, sorted best-first -- no
separate toggle needed, near-misses are simply visible alongside
qualifying pairs in the same table. A new "Analiza wybranej pary" section
below it: two ticker dropdowns (populated from the full universe, not
restricted to scan results) + an "ANALIZUJ PARĘ" button showing
color-coded pass/fail badges for all 4 gates plus a 3-panel chart
(equity curves / Z-score with entry-threshold reference lines / stacked
weight-history area) built with `plotly.subplots.make_subplots`, matching
the reference script's visual structure. Three sliders (entry Z, exit Z,
favour weight) recompute the chart live via `simulate_pair_strategy` --
fetching prices happens ONCE per "ANALIZUJ PARĘ" click (cached in
`store-relval-pair-data`, just the two selected tickers' series, not the
whole universe), so slider movement re-simulates against already-fetched
data with no additional network calls.

Verified: near-miss table rendering confirmed to include both a
fully-qualifying pair AND a 1-of-4 pair in the same result set with
correct badge/status text; `populate_pair_pickers` confirmed to list every
universe ticker; `analyze_pair` confirmed for both a valid distinct pair
and the same-ticker-twice rejection case; the 3-panel chart confirmed to
render a placeholder with no cached data, and to actually change
(non-trivial numeric difference in the plotted strategy curve) when either
`entry_z` or `favour_weight` changes -- not just that the callback fires,
that its OUTPUT numerically differs. Full app integration recheck (52/52
callbacks -- 49 prior + 3 net-new here), solver regression rerun
(identical weights to every prior check in this project's history).

---

### 7b. Real bug found and fixed — 2Y drift row-count vs calendar-date bug

The project owner reported a concrete inconsistency: the SAME pair (AVGO vs
P) showed "4/4 gates" in the universe scan table but "3/4 gates" when
analyzed individually moments later, with drift 25.1pp in one case and
62.6pp in the other (p-value/half-life/hedge-ratio only shifted slightly
between the two).

**Root cause, confirmed by reading `data/market_data.py`'s
`fetch_universe_prices`:** `prices = prices.dropna(how='all').ffill().bfill()`.
When yfinance batch-downloads MANY tickers spanning different exchange
calendars (this project's universe mixes NYSE names with Korean tickers
like "000660.KS"/"005930.KS"), the returned date index is the UNION of
every exchange's trading days -- a US ticker gets "phantom" ffilled rows on
days a foreign market was open but NYSE was closed. `engine/pairs.py`'s 2Y
drift calculation used a fixed ROW-COUNT offset (`iloc[-504]`, "504 trading
days ago"), which is only equivalent to "2 calendar years ago" if the
series has exactly 252 real trading days per year with no extra rows --
the total row count (and therefore which calendar date `-504` lands on)
depends on which OTHER tickers happened to be in the same batch request,
so the same pair fetched alone (2 tickers, clean calendar) vs. as part of
a 14-ticker universe scan pointed to genuinely different calendar dates as
"2 years ago".

**Fix:** new `_drift_diff_2y()` helper anchors the reference point to an
actual CALENDAR date (`today - CALENDAR_DAYS_2Y`, located via
`searchsorted` against the real DatetimeIndex) instead of a row-count
offset -- invariant to how many phantom rows are present, by construction.
Applied consistently in both `evaluate_pair_diagnostics` and
`scan_universe_diagnostics`'s cheap pre-filter (previously the two used
DIFFERENT drift logic paths, which itself was a latent inconsistency risk
beyond just the phantom-row issue). Verified numerically: a synthetic
"solo vs batch" reproduction (same underlying prices, one version with
extra ffilled rows injected near the 2Y boundary) confirmed the new method
gives IDENTICAL drift values regardless of phantom rows (0.0000pp
difference in two separate test constructions), while the old method is --
by definition of positional indexing -- provably sensitive to total row
count. A full regression against the existing 4-scenario synthetic test
suite (Section 5l/7a) confirmed `find_cointegrated_pairs` and
`scan_universe_diagnostics` still behave correctly after the change.

### 7c. Backtest Attribution — new second sub-tab in Relative Value (complete)

The project owner raised a deeper methodological question after reviewing
real scan output: pairs passing only 2/4 formal gates (real-world example
named: MU/WDC/SK Hynix/Samsung memory-sector pairs) produced excellent
`simulate_pair_strategy` backtest results, while a pair passing 3/4 gates
(AMD/MRVL) performed averagely or worse. This is now understood and
addressed, not just observed: the 4 screening gates test whether a
statistically stable equilibrium EXISTS (necessary for the mechanism to
make sense at all), which is a different question from whether a
Long-Only threshold-switching strategy actually PROFITS from it. A pair
can be "significantly cointegrated" with tiny, rarely-triggered spread
swings that barely beat the passive benchmark, while a pair that
technically fails the formal screen can still perform very well if its
Z-score swings are large and both names share a correlated growth trend
the Long-Only mechanism captures on top of pure mean-reversion.

**New engine functions (`engine/pairs.py`):**
- `run_backtest_batch(prices_df, pairs_df, entry_z, exit_z, favour_weight, ...)`:
  runs `simulate_pair_strategy` for every pair in a candidate table (e.g.
  `scan_universe_diagnostics`'s full near-miss output), under the same
  threshold settings, ranked by ACTUAL "Alpha vs Benchmark [pp]"
  (= `(final_strategy/final_benchmark - 1) * 100`) rather than by gate-pass
  count. Carries every original column through unchanged (so gate
  diagnostics remain available for attribution) and adds "Final Equity",
  "Max Drawdown [%]", "N Transitions", "Z-Score Std", and "Raw Spread
  Volatility" (annualized std of the daily log-price-ratio, UNNORMALIZED).
- `compute_correlations(df, target_col, candidate_cols)`: generic Pearson
  correlation of each candidate column against a target column -- the
  direct numeric answer to "which variable actually explains the
  outperformance". Deliberately agnostic to where candidate columns came
  from (gate diagnostics, `run_backtest_batch`'s own output, or a
  `data.universe_store`-derived "Same Sector" boolean attached by the UI
  layer -- `engine/` never imports `data/`, per the one-directional
  dependency rule, so sector-matching is computed in `ui/module_relative_value.py`,
  not here).

**A real, informative finding from testing (not a bug, but an important
correction to the initial design):** a synthetic 5-pair test with
KNOWN, monotonically increasing raw spread amplitude (and therefore known
increasing expected Alpha) initially showed **"Z-Score Std" correlating
NEGATIVELY (-0.826)** with Alpha -- backwards from the intended
hypothesis. Investigation confirmed this is not a bug in the correlation
mechanism (which is mechanically correct and well-defined) but a real
limitation of the chosen variable: Z-scoring is SCALE-INVARIANT by
construction, so it can look similar across pairs with very different raw
spread amplitude even though that raw amplitude is what actually
determines the dollar-magnitude of gain captured per threshold crossing.
Added "Raw Spread Volatility" (unnormalized) as a second, complementary
variable specifically to capture this -- re-verified on the same synthetic
data: correlates strongly and correctly POSITIVELY with Alpha (+0.915),
with the pair ranking now matching the known ground truth exactly
(monotonic in raw amplitude: 0.05 > 0.04 > 0.03 > 0.02 > 0.01 -> Alpha
762 > 367 > 121 > 84 > 32).

**UI (`ui/layout.py`, `ui/module_relative_value.py`):** `module-relval`
restructured into an internal `dcc.Tabs` (`relval-subtabs`) with two
sub-tabs, per the project owner's explicit preference over an
ever-growing single-page stack: "SKANER I ANALIZA PARY" (the existing
scan + manual-pair-analysis content, unchanged) and "ANALIZA WSTECZNA
(BATCH)" (new). The new sub-tab: a minimum-gates-passed filter (dropdown,
default 2/4), a "URUCHOM ANALIZĘ WSTECZNĄ" button that re-fetches +
re-scans the universe and runs the full batch backtest (self-contained,
not reusing a cross-callback price cache -- simpler, avoids Store-size
concerns, and the marginal re-scan cost is small relative to the batch
backtest itself, which is already a multi-minute operation for a large
universe), a correlation summary (horizontal bar per candidate variable,
colored green/red by sign, width by magnitude) reading directly from
`compute_correlations`'s output, and a results table sorted by actual
Alpha with a "Sektor" column (Same/Different) for the sector-match
hypothesis.

Verified: the full pipeline was tested end-to-end against a synthetic
scenario deliberately constructed to reproduce the project owner's
real-world observation -- a "MEMORY"-style pair (weaker formal
cointegration, large spread amplitude, strong shared growth trend, same
sector) vs. a "STABLE"-style pair (textbook-strength cointegration, small
spread amplitude). Result: the weaker-gates pair (3/4) produced
Alpha +705.2pp; the fully-qualifying pair (4/4) produced only
+16.3pp -- directly reproducing the reported phenomenon, confirming the
tool surfaces exactly the effect it was built to investigate. Correlation
bar rendering confirmed structurally correct (8 candidate variables
rendered, though with only 2 pairs in this specific test every correlation
is mathematically forced to exactly +-1.0 -- expected behavior for n=2,
not a bug, a real universe scan with many pairs will show a genuine
spread of values). Full app integration recheck (53/53 callbacks -- 52
prior + 1 net-new here), solver regression rerun (identical weights to
every prior check in this project's history).

---

### 7d. 2Y Drift Gate Removed; Relative Alpha and Composite Ranking Redefined (complete)

Follow-up to 7c: the project owner reviewed real batch-attribution output
(screenshot) and confirmed empirically what was suspected -- "Drift Diff 2Y
[pp]" showed essentially zero correlation with actual backtest Alpha
(+0.021, the weakest of every candidate variable). Root cause, now
understood precisely: the metric was a single POINT-IN-TIME comparison
(today vs. exactly 2 years ago) in ABSOLUTE percentage-point units, applied
across a universe where individual stocks' 5-year total returns range from
roughly +400% to +2500% -- two price paths that moved together closely the
entire time can still show a huge point-to-point percentage-point gap
purely because one stock's absolute compounding scale happened to be much
larger, which has nothing to do with whether the paths actually diverged
from each other.

**Removed entirely, not just de-weighted:** `_drift_diff_2y`,
`CALENDAR_DAYS_2Y`, `DEFAULT_MAX_DRIFT_PP`, the `max_drift_pp` parameter
threaded through `evaluate_pair_diagnostics`/`scan_universe_diagnostics`/
`find_cointegrated_pairs`, and the cheap drift PRE-FILTER in
`scan_universe_diagnostics` (previously a hard exclusion before the
expensive cointegration test even ran). **Formal gate count dropped from 4
to 3** (cointegration p-value, half-life, hedge-ratio sanity) everywhere:
`gates_passed`/`all_passed` semantics, the "X / N" UI labels (now "/ 3"),
and the batch tab's minimum-gates dropdown (now offers 1-3, not 1-4).
Confirmed explicit consequence, not an oversight: universe scans are now
slower on a large universe (no early-exit at all on any gate before the
full cointegration test runs) -- an accepted tradeoff, since the removed
gate wasn't actually predictive of anything worth optimizing scan speed
around.

**New function, `avg_relative_divergence_5y(prices_a, prices_b)`:**
rebases both price paths to 100 at the start of the (however long) window
provided, then averages `|ln(norm_A(t)) - ln(norm_B(t))|` across EVERY day
in that window (not one point-in-time snapshot). Symmetric (order of A/B
doesn't matter) and expressed in the same log-price units as the
cointegration spread and hedge-ratio regression. Reported PURELY as an
informational field in `evaluate_pair_diagnostics`'s output (does not
participate in `gates_passed` at all) -- confirmed explicit design: "chcę
najpierw zobaczyć jaką będzie miał korelację, później zobaczymy może
ustalimy jakiś sztywny próg" (see this variable's own correlation reading
once real batch runs accumulate, before considering a fixed threshold).
Verified with four targeted numeric tests: exactly 0 for two identical
paths; exactly matches a known analytically-derived value (a linear
log-ratio ramp from 0 to X averages to X/2, confirmed to 1e-9); symmetric
under swapping A and B; and invariant to multiplying both price series by
a constant scale factor (confirmed identical to 1e-9) -- directly verifies
this measures PATH divergence, not absolute price-level differences,
which is exactly what the removed metric got wrong.

**"Relative Alpha [%]" redefined** (`run_backtest_batch`, was "Alpha vs
Benchmark [pp]"): now divides by `max(final_benchmark, final_100a,
final_100b)` -- the best of the three NON-dynamic curves, whichever that
happens to be -- rather than always dividing by the passive 50/50
benchmark specifically. The underlying ratio-based formula itself was
already correctly relative (not a raw arithmetic percentage-point
difference, despite the old "[pp]" label suggesting otherwise) -- verified
against the project owner's own two worked examples before writing any
code: (2000/1500 - 1)*100 = 33.3% and (500/335 - 1)*100 = 49.25% (≈"50%"),
both confirmed to match exactly. Only the denominator changed, plus a
clearer "[%]" label replacing the misleading "[pp]" one.

**New "Composite Score"** (confirmed ranking formula): 50% each of
"Relative Alpha [%]" and "Days In Lead [%]" (already existing from 7b),
combined via PERCENTILE RANK (`pandas.Series.rank(pct=True)`) rather than
a raw weighted sum -- a deliberate choice, not arbitrary: Alpha and Days
In Lead live on very different numeric scales (Alpha can range from
roughly -90% to well over +1000%; Days In Lead is bounded to [0, 100]), so
a raw `0.5*Alpha + 0.5*DaysInLead` would be completely dominated by
whichever metric has the larger range in a given batch. Rank-based
combination is scale-invariant: each half of the score reflects the
pair's RELATIVE STANDING within that batch, not its raw magnitude, so both
genuinely carry equal weight. Verified by manually recomputing the
percentile ranks and composite score outside the function and confirming
an exact match against `run_backtest_batch`'s own output. Batch results
now sort by Composite Score descending, not by Alpha alone.

**Full pipeline re-verified end to end** with a targeted scenario
reproducing the exact failure mode that motivated this whole change: a
pair with drastically different 5-year absolute returns (+141% vs. +67%, a
74pp gap -- which WOULD have been hard-rejected by the old 35pp drift
gate before ever reaching the cointegration test) now correctly appears in
results (2/3 gates passed) and shows strong actual performance
(Relative Alpha +451.8%, Days In Lead 96.8%) -- directly confirming the
fix addresses the real-world case the project owner observed with
memory-sector pairs being silently excluded despite strong backtest
potential. Full app integration recheck (53/53 callbacks -- same count,
this was a redefinition of existing callbacks' internals, not new
callback registrations), solver regression rerun (identical weights to
every prior check in this project's history), and a project-wide grep
confirming zero remaining references to any removed drift-gate identifier
(`drift_diff_pp`, `drift_pass`, `Drift Diff 2Y`, `max_drift_pp`) or the old
Alpha column name (`Alpha vs Benchmark [pp]`) anywhere in `engine/` or `ui/`.

---

## 8. Implementation Status & Development Roadmap

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

## 9. Execution Instructions

### Prerequisites:
* Python `>= 3.10`
* Dependencies listed in `requirements.txt`: