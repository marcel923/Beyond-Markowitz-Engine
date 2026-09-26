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

### 7e. Walk-Forward Out-of-Sample Validation — third Relative Value sub-tab (complete)

Confirmed motivation (2026-09-08, fifth follow-up): everything built for
Relative Value so far (screening, batch attribution) was purely
IN-SAMPLE -- a pair was discovered and immediately evaluated on the SAME
data used to find it, which cannot distinguish a genuine, persistent
relationship from a pattern that happened to fit one specific historical
window. This adds a walk-forward split: 5 years TRAIN + 1 year TEST
(confirmed lengths, chosen to align with the project's existing
~21-trading-day/monthly rebalance cadence -- a 1-year OOS window gives
roughly 12 rebalance cycles' worth of validation).

**Design, confirmed point by point:**
1. **Split boundary is a CALENDAR date**, not a row-count offset --
   `prices.index[-1] - pd.DateOffset(years=test_years)` -- deliberately
   reusing the exact lesson from the earlier 2Y-drift row-count bug (a
   fixed row-count split lands on a different calendar date depending on
   how many "phantom" ffilled rows a mixed-exchange batch fetch produces).
2. **Hedge ratio and every gate diagnostic are computed ONLY from TRAIN**
   (`evaluate_pair_diagnostics` called on the TRAIN slice alone) -- the
   whole point of the exercise fails if the "unseen" test year is allowed
   to influence which parameters get used to evaluate it.
3. **The backtest itself runs across the FULL window in one pass** (not
   simulated separately per half) using the TRAIN-derived hedge ratio
   throughout, so the rolling Z-score at the start of the TEST year still
   has its full lookback of genuine preceding data -- exactly like a live
   system crossing that date would experience, not an artificially cold start.
4. **The resulting single equity curve is then split into IS and OOS
   segments**, with the OOS segment REBASED to 100 at the boundary (each
   curve divided by its own value on that day) -- isolates what happened
   ONLY in the held-out year, so a pair that was merely lucky in-sample
   can't look artificially good out-of-sample purely from carried-forward
   compounding.
5. **Composite Score comparison, confirmed exact request** ("chcę różnicę
   score 50/50 zobaczyć jak się zmienił"): "Composite Score (IS)" and
   "Composite Score (OOS)" are each computed via percentile-ranking every
   candidate pair in the batch on ITS OWN segment's Alpha/Days-In-Lead --
   two INDEPENDENT rankings, not one reused for both -- so "Composite
   Score Δ (OOS - IS)" answers "did this pair's relative standing among
   its peers hold up," not just whether its raw numbers moved. No fixed
   pass/fail threshold was added (explicitly declined in favor of showing
   the actual delta) -- the project owner can eyeball which pairs held up
   directly from the sorted table.
6. **Confirmed addition** ("możesz dodać"): a separate statistical
   re-test of cointegration on the OOS price slice alone, with hedge ratio
   still FIXED at its train-derived value. Because gamma is a known
   constant here rather than something being jointly estimated from the
   tested data, the methodologically correct test is a PLAIN Augmented
   Dickey-Fuller test on the resulting fixed-gamma spread
   (`statsmodels.tsa.stattools.adfuller`), NOT the two-step Engle-Granger
   `coint()` used for the original screen -- `coint()`'s MacKinnon-adjusted
   critical values specifically correct for jointly estimating the
   regression coefficient on the same data under test, which does not
   apply once gamma is already fixed. "Gates Passed (OOS)" is therefore
   scored out of 2 (cointegration + half-life), not 3 -- hedge-ratio
   sanity isn't re-tested since it's unchanged by construction.
7. **Data source**: self-contained, matching the established pattern for
   the other two sub-tabs -- fetches 6Y fresh (5Y train + 1Y test) rather
   than reusing a cross-callback cache (owner's call: "jak uważasz będzie
   lepiej" -- consistency with the existing pattern was judged better than
   introducing a new caching mechanism for one tab).

**New engine function, `run_walk_forward_validation(prices_df, pairs_df,
test_years, ...)`** in `engine/pairs.py`. A real, if minor, forward-compatibility
fix made in passing: `statsmodels.tsa.stattools.adfuller` emitted a
`FutureWarning` about its return-value shape changing in a future release
-- pinned explicitly via `result_object=False` to silence it and lock in
the current tuple-based behavior this code already expects, rather than
letting a future statsmodels upgrade silently change behavior underneath it.

**UI**: third `dcc.Tab` ("WALIDACJA OUT-OF-SAMPLE") in the Relative Value
module, alongside a minimum-gates filter (evaluated on the full window, to
choose which candidates are worth walk-forward testing at all) and a
test-period-length input (default 1 year, confirmed). Results table sorted
by Composite Score (OOS) descending, with the Δ column colored
green/red by sign for quick visual scanning.

Verified with two deliberately contrasting synthetic scenarios, run
through the full UI callback (not just the engine function directly): a
GENUINELY persistent pair (same underlying cointegration dynamics for the
entire 6-year window) and a pair whose relationship BREAKS DOWN precisely
at the train/test boundary (independent, diverging trends injected only in
the final year). Result: the persistent pair correctly ranks best
out-of-sample (Composite Score Δ +0.083, OOS score 1.000 -- the top of the
batch); the breaking pair, despite a PERFECT 3/3 gates in training (it
looked completely qualified in-sample), shows a clearly NEGATIVE delta
(-0.333, OOS score dropping to 0.583) -- directly demonstrating the tool
catches exactly the failure mode it was built to catch: a pair that looks
great purely because it was evaluated on the same data used to find it.
Full app integration recheck (54/54 callbacks -- 53 prior + 1 new), solver
regression rerun (identical weights to every prior check in this
project's history).

---

### 7f. Monthly Rolling Walk-Forward — production-mechanism prototype (complete)

Confirmed direction change (2026-09-08, sixth follow-up): the project
owner explicitly rejected cointegration p-value as something that matters
at all -- "nie obchodzi mnie jaka jest p-value... interesuje mnie
najwyzszy score i w jakim horyzoncie bedzie sie to utrzymywalo
STATYSTYCZNIE." This directly follows from Etap 7c's own finding
(P-Value correlated NEGATIVELY with actual backtest Alpha) -- continuing
to treat p-value as central after finding that would have been
inconsistent. From this point on, a pair's edge is judged purely by
whether ITS OWN realized performance is statistically distinguishable
from zero, not by any cointegration "quality" measure. hedge_ratio is
kept only as the technical device needed to construct the spread/Z-score
-- it carries no claim about statistical validity on its own anymore.

**This is a genuinely different mechanism from everything built so far**,
not another variant of the discrete daily threshold-switching backtest
(`simulate_pair_strategy`): a smooth, bounded weight tilt
(`weight_A = 0.5 + 0.5*theta*tanh(-Z)`), checked ONCE per ~21-session
rebalance cycle and held with zero within-month reaction -- a direct
prototype of the eventual production mu_i-nudge mechanism, expressed here
in weight-space (since this is an isolated 2-asset diagnostic, with no
portfolio-level mu_i to nudge) rather than the literal production
mechanism itself.

**Why the Z-score baseline window is 252 sessions, not 63** (confirmed,
explained to the project owner before coding): (1) both the mean and std
used to Z-score the spread are themselves noisy ESTIMATES, with standard
error scaling as `1/sqrt(n)` -- a monthly-checked signal gets no chance to
average that estimation noise out between checks the way a daily-reacting
system would, so the reference point needs to be more stable to begin
with; (2) this project's own half-life gate tolerates reversion cycles up
to 63 sessions -- using a 63-session window to estimate the "baseline"
risks spanning LESS than one full cycle, contaminating the estimate with
wherever in the oscillation the window happens to sit rather than its true
center. 252 sessions comfortably spans several full cycles even at the
slow end of the tolerated range.

**Two real, instructive mistakes found and fixed while verifying this
function** (both worth recording in full, since they're genuine lessons
about the domain, not just fixed typos):

1. **A real bug**: the "current spread" at each month's start was computed
   as `log(P_A) - hedge_ratio*log(P_B)`, OMITTING the regression intercept
   (alpha) that `_hedge_ratio_and_spread` DOES include when constructing
   the baseline spread series it returns. This created a constant
   systematic offset between the "current" point and its own baseline,
   producing absurd, persistently saturated Z-scores (+6 to +14, pinned at
   the tanh saturation bound every single month) -- caught immediately
   because the resulting weights never varied. Fixed by using the
   baseline spread series' own last value (`train_spread.iloc[-1]`)
   directly instead of recomputing it separately with a different,
   inconsistent formula.
2. **A test-construction mistake, not a code bug, but informative**: an
   initial synthetic "genuinely persistent" test pair used an AR(1) spread
   with phi=0.983 (nominal half-life ~40 sessions) that turned out to sit
   too close to a unit root -- such a process's UNCONDITIONAL variance
   grows large and its short-run trajectory can drift far from any
   252-day baseline for extended stretches, since near-unit-root
   processes take many multiples of their own half-life to reveal a
   stable long-run center. This reproduced the same saturated-Z-score
   symptom as bug #1, but for a different, purely statistical reason,
   distinguished by direct diagnostic comparison against a hand-computed
   Z-score. A second test attempt used a smooth, deterministic
   sinusoidal spread, which turned out to be an inappropriate model
   entirely: a pure sine wave's LEVEL and its DERIVATIVE (which is what
   determines the next period's forward return) are exactly 90-degrees
   phase-shifted and therefore have ZERO correlation by construction
   (confirmed numerically to machine precision) -- unlike a true AR(1)
   mean-reverting process, where the level directly determines the
   expected direction of the next move. Both attempts were abandoned in
   favor of a properly-scaled AR(1) process (phi=0.90-0.95, an
   unconditional variance kept moderate by choice of sigma), which
   correctly validated the mechanism.

**Verification, once a proper test model was used**: a direct diagnostic
(bypassing the theta/tanh weight formula entirely) confirmed Z-score at
month start correlates with the REALIZED next-month return differential
`(ret_A - ret_B)` at +0.734 for a well-scaled AR(1) pair -- the core
signal is directionally sound. The full 12-month AGGREGATE t-statistic
for that same single random realization did not reach conventional
significance (t=0.91, p=0.38) -- confirmed, on inspection, to be an
honest consequence of small-sample randomness (only 2 of the 12
non-overlapping months happened to show a large Z-score in that specific
draw, diluting the aggregate with 10 near-zero-signal months) and the
deliberately conservative `theta=0.15` (a gentle tilt, per the project's
Long-Only, non-arbitrage philosophy), not a flaw in the mechanism -- this
is precisely the caveat the t-statistic is supposed to surface: a
genuinely-present signal is not automatically the same as slam-dunk
statistical proof at n=12. The batch UI status message says this
explicitly ("przy n=12 to surowy próg -- traktuj jako wstępny filtr, nie
ostateczny dowód").

**New engine functions**: `simulate_monthly_walkforward` (one pair,
returns per-month details plus mean/std/t-stat/p-value across the 12
months) and `run_monthly_walkforward_batch` (runs it across every
candidate pair, sorted by `|t-statistic|` descending -- confirmed priority:
the t-statistic is the single number that combines both the SIZE of a
pair's average edge and the CONSISTENCY of it, which is exactly "highest
score, over what horizon does it hold up statistically").

**UI**: fourth `dcc.Tab` ("TRWAŁOŚĆ MIESIĘCZNA (t-TEST)") in the Relative
Value module, with a minimum-gates filter (coarse candidate selection only
-- p-value plays no further role), a `theta` input (default 0.15,
adjustable for experimentation as explicitly invited), and a results table
sorted by t-statistic with p<0.05 rows highlighted.

Verified end to end through the full UI callback (not just the engine
functions directly) on synthetic data; full app integration recheck
(55/55 callbacks -- 54 prior + 1 new), solver regression rerun (identical
weights to every prior check in this project's history).

**Explicitly NOT yet done** (per the project owner's own sequencing,
confirmed this same follow-up): the actual pair-assignment mechanism for
wiring this into the Stage 1 SLSQP solver -- "kazda spolka moze byc
przydzielona tylko do jednej pary ale to na pozniej, najpierw musze uzyskac
taki wynik ktory da mi mozliwosc potwierdzenia ze obliczenia sprawdzaja sie."
This tab produces the confirmation tool; solver wiring is an explicit,
separate, later step.

---

### 7g. Chart-first IS/Trailing Score view; p-value fully removed from this tab (complete)

Confirmed frustration and redirect (2026-09-08, seventh follow-up): the
project owner found the Etap 7e OOS table (screenshot reviewed) unreadable
as a wall of numbers with no visual -- "kompletnie nie wiem na co mam sie
patrzec... nie ma zadnych wykresow". Also confirmed: cointegration p-value
is no longer trusted as informative at all here ("nie wiem czy p-value
jest tu w ogole przydatne... ciagle gdzies je wrzucasz") -- it has been
removed ENTIRELY from this tab's chart and table (it still exists
elsewhere in the module, e.g. the original scan table, where it remains a
coarse candidate filter, just not a headline metric here).

**The project owner's own proposed metric, now implemented**: take a
pair's Composite Score computed on the 5-year IS/training window (already
existing -- reused directly from `run_walk_forward_validation`'s
"Composite Score (IS)", no new compute needed), and separately compute a
TRAILING 12-month analysis specifically using the theta/monthly production
mechanism (confirmed: "teraz tylko trzeba to zaimplementowac dla strategii
z theta") to get both a mean score AND its standard deviation -- directly
the two numbers requested ("ranking ze scorem i odchyleniem tego score").
The project owner's own empirical observation from the prior table (pairs
scoring above ~0.8 in BOTH windows tended to beat the best single-stock
alternative) is now the organizing principle of this view, made visual
rather than left as something to notice buried in a dense table.

**New engine function, `run_theta_trailing_stability`**: for every
candidate pair, runs `simulate_monthly_walkforward` to get its trailing
12 raw monthly alphas, then -- the key methodological step -- for EACH of
those 12 months independently, percentile-ranks EVERY candidate pair's
alpha for THAT SPECIFIC MONTH against every other candidate pair (not
each pair scored against its own history in isolation). This gives each
pair a 12-point time series of cross-sectional "monthly scores" (0-1,
same convention as Composite Score elsewhere); "Trailing Score" is the
mean of that series, "Trailing Score (std)" is its standard deviation --
exactly "score i odchylenie tego score", now specifically for the theta
mechanism. Pairs are included only with a COMPLETE 12-month history (a
fair monthly cross-sectional ranking needs every included pair to have a
value for every compared month).

Verified on three deliberately different synthetic pairs (a moderate,
well-scaled AR(1) "CONSISTENT" pair; a high-amplitude "ERRATIC" pair;
a low-amplitude "WEAK" pair): the erratic pair scored both the highest
mean AND the highest std (matches Etap 7f's own finding that raw
amplitude drives captured alpha, for better or worse), the weak pair
scored both the lowest mean and a comparatively tighter std -- the
function differentiates pairs sensibly along both dimensions, not just
one.

**Combining two DIFFERENT mechanisms, made explicit rather than silently
mixed**: "IS Score" (x-axis) comes from the discrete threshold-switching
backtest over the full 5-year training window; "Trailing Score" (y-axis)
comes from the theta/monthly mechanism over the trailing 12 months. This
is a deliberate, documented choice, not an oversight -- computing a
theta-based IS score over the training window would itself require
several YEARS of data preceding that training window (each of its own
sub-months would need its own 5-year lookback), which isn't practical
given this project's existing 6-year fetch. The two axes answer
genuinely different, complementary questions ("was this pair generally
good historically, by any reasonable method" vs. "does the actual
production-candidate mechanism show a stable, consistently well-ranked
edge recently") rather than pretending to be the same measurement twice.

**UI**: the fourth sub-tab ("TRWAŁOŚĆ MIESIĘCZNA (SCORE + WYKRES)") is now
chart-first. An error-bar scatter (`plotly.graph_objects`, matching the
project's existing chart conventions) plots IS Score vs. Trailing Score
(mean) with error bars showing Trailing Score (std), reference lines at
the empirically-observed 0.8 threshold on both axes, and a distinct
marker color for pairs clearing both -- the pattern the project owner
described ("wezmiesz spolki ktore w obu przypadkach maja powyzej 0.8
score") is now something to SEE (upper-right cluster with short error
bars) rather than something to notice by scanning columns. A detail table
below the chart carries the same three numbers (IS Score, Trailing Score
mean, Trailing Score std) for sorting/filtering -- with NO p-value column
anywhere in this tab.

Verified end to end through the full UI callback (not just engine
functions directly) on synthetic data -- confirmed the merged
DataFrame, the chart's trace count and hover data, and the table's sort
order all behave correctly together. Full app integration recheck (55/55
callbacks -- same count, this modifies an existing callback's Outputs/body
rather than registering a new one), solver regression rerun (identical
weights to every prior check in this project's history).

---

### 7h. Per-pair theta chart in Tab 1 (complete) — the small change requested first

Confirmed feedback (2026-09-08, eighth follow-up): the Etap 7g
cross-sectional scatter (547 real pairs at once) looked like pure
randomness to the project owner -- correctly so, since a scatter of
hundreds of points conveys population-level structure, not what any ONE
pair's mechanism actually does over time. Explicit request, taken as a
deliberately small first step before revisiting whether 12 months is the
right trailing window at all ("zacznijmy od tej małej zmiany a później
się zastanowimy"): add a per-pair chart of the theta/monthly mechanism
(not the discrete 80/20-style one) to the existing single-pair analysis
section in Tab 1 ("Skaner i Analiza Pary").

**New engine function, `simulate_monthly_theta_curve`**: the daily-resolution
visual counterpart to `simulate_monthly_walkforward` (which only returns
12 summary numbers) -- same underlying mechanism (weight tilted via
`0.5 + 0.5*theta*tanh(-Z)`, re-evaluated once per ~21-session month using
a 252-session baseline from the preceding 5-year window), but returning a
full day-by-day equity curve across the whole test window so a single
pair's actual behavior can be plotted, the same way the discrete
threshold chart already lets a person inspect one pair. Physical-share
execution (same anti-"Shannon's Demon" approach as `simulate_pair_strategy`):
shares are only re-set at each MONTH boundary, drifting naturally with
prices in between.

**A test-threshold mistake caught and corrected while verifying this
(not a code bug)**: an initial sanity check asserted weight changes
within a month must stay below an arbitrary absolute value (0.01), which
failed at a measured 0.0138 -- investigation confirmed this is CORRECT,
expected behavior, not a bug: with share counts frozen for the whole
month, the realized weight naturally drifts as each stock's own price
moves, exactly as intended (that drift is what avoiding Shannon's Demon
looks like). The test was rebuilt as a RELATIVE comparison instead of an
arbitrary absolute threshold: mean day-to-day weight change at month
boundaries (where the real rebalance happens) was confirmed to be ~18x
larger than mean day-to-day change within months (natural price drift
only) -- the correct qualitative signature, verified quantitatively
rather than eyeballed. Also verified: no Shannon's Demon (50/50 benchmark
stays within the two single-stock curves' corridor), and exact numerical
agreement between this function's per-month weights and
`simulate_monthly_walkforward`'s own weights for the same pair and
parameters (both derive from the identical calculation, cross-checked
independently rather than assumed).

**UI**: a new panel directly below the existing threshold-based chart in
Tab 1's "Analiza Wybranej Pary" section, for the SAME currently-selected
pair (reuses `store-relval-pair-data`, no extra fetch) -- a 2-panel chart
(equity curves on top; weight-in-portfolio below, annotated with vertical
dotted lines at each month boundary) with its own `theta` and
"liczba miesięcy" inputs, independent of the discrete mechanism's
entry/exit/favour-weight sliders since this is a structurally different
mechanism. Verified end to end (placeholder with no pair selected;
full chart with correct trace names once a pair is analyzed; changing
theta produces a measurably different curve).

Full app integration recheck (56/56 callbacks -- 55 prior + 1 new), solver
regression rerun (identical weights to every prior check in this
project's history).

**Explicitly deferred, per the project owner's own sequencing**: whether
to shorten the trailing window from 12 months to 2-3 (since each month
already gets a fresh 5-year retrain regardless) is an open question to
revisit once this per-pair visual has been used to build intuition first
-- not decided or changed in this pass.

---

### 7i. Theta Batch Attribution — fifth sub-tab, mirroring Tab 2 (complete)

Confirmed observation (2026-09-08, ninth follow-up): running the theta
mechanism pair-by-pair (Etap 7h's chart), the project owner found that
memory-sector pairs (WDC/000660.KS and similar) show no edge under this
mechanism, while pairs passing all 3 gates show a modest but apparently
REPEATABLE edge that scales up with theta. Explicit request: build a
batch-level equivalent of Tab 2's "Analiza Wsteczna (Batch)" -- same
correlation-bar + scatter + ranked-table structure -- but powered by the
theta/monthly mechanism instead of the discrete threshold one, with theta
itself adjustable, to test this observation systematically across every
scanned pair rather than one at a time.

**Engine fix required first**: `run_monthly_walkforward_batch` previously
discarded every column from its input `pairs_df` except Ticker A/B,
carrying through none of the gate diagnostics (P-Value, Half-Life, Hedge
Ratio, Avg Relative Divergence, Gates Passed) needed for correlation
analysis. Fixed to carry through all original columns unchanged, via
`dict(row)`, matching the pattern already used by `run_backtest_batch`.
Verified directly: gate-diagnostic columns are now present and populated
correctly in the function's output.

**A genuine, mathematically confirmed finding from testing, worth
recording**: increasing theta from 0.15 to 0.40 (a 2.67x increase) scaled
"Śr. Alpha Miesięczna [pp]" by almost exactly the same factor (2.666x),
while the t-statistic stayed EXACTLY unchanged (2.901 in both runs). This
is not a coincidence -- it follows directly from the mechanism's linearity
in theta: `weight_A - 0.5 = 0.5*theta*tanh(-Z)` scales every month's
realized alpha by the same constant factor, which scales BOTH the mean and
the standard deviation of the 12 monthly alphas proportionally, leaving
their ratio (the t-statistic) invariant. Practical implication, stated
explicitly to the project owner: increasing theta amplifies the SIZE of
the edge (and proportionally, the risk/drawdown) but does NOT increase
confidence that the edge is real -- a decision to raise theta is a
decision to bet more on the same signal, not evidence of a better signal.

**UI**: fifth `dcc.Tab` ("ANALIZA WSTECZNA THETA (BATCH)"), structurally
mirroring Tab 2 -- minimum-gates filter, adjustable `theta`, a "run"
button, correlation bars (against "Śr. Alpha Miesięczna [pp]", the
theta-mechanism's own outcome measure -- explicitly NOT the same quantity
as the discrete mechanism's "Relative Alpha [%]", called out in the UI
copy so the two are not confused as comparable across tabs), a p-value-vs-alpha
scatter (log-x, same convention as Tab 2's scatter, for visual
consistency), and a ranked table sorted by `|t-statistic|` descending.

Full app integration recheck (57/57 callbacks -- 56 prior + 1 new), solver
regression rerun (identical weights to every prior check in this
project's history).

---

### 7j. Half-Life Hypothesis Chart, Cumulative Return, and % Months Above 50/50 — Tab 5 only (complete)

Confirmed observation (2026-09-08, tenth follow-up): all correlations in
Tab 5 (theta batch attribution) came back negative against
`Śr. Alpha Miesięczna [pp]`, including Half-Life at -0.347 (the
strongest), yet the project owner separately observed in Tab 1's per-pair
chart that fully-qualifying pairs looked consistently good above 50/50.
Two candidate explanations were proposed and discussed before any code
change, per the project owner's explicit request to think it through
first: (1) the same near-unit-root failure mode already documented when
this mechanism was first built (Etap 7f) -- a half-life near the gate's
upper bound (63 sessions) may be too close to a unit root for the
252-session baseline to capture a genuinely stable equilibrium; (2) a
modest but consistent monthly edge can compound into a visually
convincing cumulative equity curve (what Tab 1 shows) while still not
clearing a strict month-to-month t-test at n=12 (what Tab 5's correlation
table reflects) -- these are different questions, not contradictory
findings.

**Explicitly scoped to Tab 5 only, per direct instruction** ("pamietaj co
do punktu 3 ze to ma isc do zakladki 5 a nie do drugiej, drugiej nie ruszaj
niech bedzie jaka jest") -- Tab 2 (the discrete-mechanism batch attribution)
is completely untouched by this change.

**New engine outputs, `simulate_monthly_walkforward` and
`run_monthly_walkforward_batch`**: two additional summary statistics per
pair, addressing hypothesis #2 directly:
- `cumulative_alpha_pp`: the TRUE compounded return of the theta-tilted
  allocation across all usable months, minus the compounded 50/50 return
  over the same months -- deliberately NOT `mean_alpha * n_months`, since
  compounding is non-linear and order-dependent; verified numerically that
  the naive product-of-average differs measurably (by ~0.32pp in one test
  run) from the correctly compounded figure.
- `pct_months_positive`: % of usable months where the tilted allocation
  beat 50/50 for that month alone -- the direct monthly analogue of "Days
  In Lead" from the discrete-mechanism tabs, explicitly measured against
  50/50 specifically (clarified in this same conversation: this tab's
  Alpha was ALREADY computed against 50/50, not "best of three
  alternatives" as in Tab 2 -- that distinction only ever applied to Tab 2).

Both verified against hand-computed reference values (compounding via
`numpy.prod`, positive-month count via direct enumeration) before being
trusted in the UI.

**New chart addressing hypothesis #1**: a Half-Life vs. t-statistic
scatter, added alongside the existing p-value scatter in Tab 5. Tested
directly against two purpose-built synthetic scenarios (a short,
252-window-appropriate half-life ~12 sessions vs. two long,
near-gate-boundary half-lives ~55-60 sessions): the short-half-life pair
showed the best t-statistic (+1.023) of the batch, while every
long-half-life pair scored markedly weaker (+0.546 down to -0.016) --
directionally consistent with hypothesis #1, though this was a small,
single-draw illustrative check (n=4 pairs), not a statistically
conclusive test; the chart now lets the project owner examine this
pattern directly against the real universe's actual pairs.

**UI**: two new table columns ("Zwrot Skum. 12M [pp]", "% Miesięcy >
50/50") and the new Half-Life vs t-statistic chart, all in Tab 5 only,
with brief in-UI copy stating the hypothesis being tested. Full app
integration recheck (57/57 callbacks -- same count, this modifies an
existing callback's Outputs/body), solver regression rerun (identical
weights to every prior check in this project's history).

---

### 7k. Extended Test Window (Option A) and Pooled Significance Test (Option B) (complete)

Confirmed follow-up (2026-09-08, eleventh follow-up) after reviewing
Tab 5's real correlation output (all negative, strongest -0.347 on
Half-Life) alongside a genuine question: is 12 months simply too few
independent observations per pair for the t-test to reliably detect a
real (if modest) edge, given that each month's realized return
differential is dominated by largely-independent, idiosyncratic
stock-return noise? Answered honestly before any code change: standard
error scales as `1/sqrt(n)`, so going from 12 to 48 months (4x) roughly
halves estimation noise -- a real, but modest improvement, not a
guarantee of newfound significance. If the true per-pair effect is small,
more months will more reliably measure that small effect, not manufacture
a large one; a still-noisy result after 48 months is itself a valid,
informative answer, not evidence anything is broken.

**Option A — longer per-pair test window, confirmed cap: 10 years total /
48 months.** No engine changes needed (`simulate_monthly_walkforward` and
related functions already accepted `n_months`/`train_years` as parameters
with the maximum simply not exposed in the UI). Changes: Tab 1's existing
"LICZBA MIESIĘCY" input max raised from 24 to 48; new n_months inputs
(max 48) added to Tab 4 and Tab 5, threaded through to
`run_theta_trailing_stability` and `run_monthly_walkforward_batch`
respectively; data fetch period raised from 5y/6y to 10y everywhere this
monthly mechanism is used (Tab 1's `analyze_pair`, Tab 4, Tab 5) --
Tab 2 and Tab 3 (the discrete-mechanism tabs) are explicitly UNCHANGED,
per direct instruction ("drugiej nie ruszaj niech bedzie jaka jest").
Tickers with less than ~9-10 years of real history (recent IPOs) simply
yield fewer usable months via the existing per-month data-sufficiency
check -- already handled gracefully, no new logic needed.

**Option B — pooled cross-pair significance test, confirmed scope: the
SAME gate-filtered candidate set already shown in the per-pair table, not
the unfiltered universe** ("dokładnie te same, przefiltrowane pary...
biorąc dokładnie te same, przefiltrowane pary"). `run_monthly_walkforward_batch`
now returns a tuple `(per_pair_df, pooled_monthly_alphas)` -- the pooled
list collects every individual (pair, month) raw alpha value from every
pair already included in the per-pair table, gathered inside the SAME
loop that already runs the backtest (no duplicate computation). New
function `compute_pooled_significance` runs one t-test on this pooled
array. Verified with a purpose-built demonstration (8 synthetic pairs,
each individually weak): only 2 of 8 pairs individually reached p<0.05,
while the pooled test across all 96 observations gave t=5.082,
p≈0.00000 -- a clean, concrete illustration of exactly the statistical-power
argument being made (this answers "does the mechanism have a systematic
effect across this set", a different and complementary question to any
one pair's own significance, not a replacement for the per-pair ranking).

**UI**: new "TEST ZBIORCZY" panel in Tab 5, positioned between the status
message and the correlation bars, explaining the distinction in plain
terms before showing the pooled n / mean / t-statistic / p-value.

Full app integration recheck (57/57 callbacks -- same count, this
modifies existing callbacks' Outputs/State/body rather than registering
new ones), solver regression rerun (identical weights to every prior
check in this project's history).

---

### 7l. Cross-Tab Consistency Audit — Screening-Window Mismatch, Stale Label, and a Correlation-Breaking `inf` (complete)

Confirmed report (2026-09-08, twelfth follow-up): Tab 1's universe scan
showed 19 pairs passing all 3 gates out of 630 surviving the drift
pre-filter, but Tab 5's batch (same universe, same day, min_gates=3)
found only 5. Explicit request: audit the WHOLE module for inconsistencies,
not just this one, since "gdzieś już idziemy ale wszystko się rozjeżdża."

**Root cause, found and confirmed, not guessed at:** Etap 7k raised the
price-fetch window for Tabs 3, 4, and 5 from 5-6 years to 10 years, to
support up to 48 test months for the theta mechanism. But those same
tabs ALSO run `scan_universe_diagnostics` (the gate screen: P-Value,
Half-Life, Hedge Ratio, Gates Passed) directly on that SAME extended
fetch -- meaning the gate-screening test itself silently started running
on a longer window than Tab 1's, which still screens on 5 years. A
cointegration test over a longer window is inherently a STRICTER test (a
relationship has to hold up over more time to pass) -- so "3/3 bramek"
quietly stopped meaning the same thing across tabs. Confirmed directly, not
inferred: a synthetic pair genuinely cointegrated only in its most recent
5 years (independent, unrelated dynamics in the preceding 5) scored 0/3
gates when screened on the full 10-year window, but 3/3 when screened on
either a fresh 5-year fetch OR the last-5-years slice of the same 10-year
fetch -- an exact, reproduced match of the reported 19-vs-5 discrepancy's
mechanism. Tab 3 (`run_oos_validation`) turned out to have carried a
version of this same class of bug since Etap 7e (it already screened on
its own 6-year fetch, not 5 years) -- caught only now via this
comprehensive pass, not previously noticed.

**Fix:** new shared helper `_screening_slice(prices_df)` in
`ui/module_relative_value.py` -- slices the last `GATE_SCREENING_YEARS`
(5, a module-level constant) calendar years from whatever longer price
history was already fetched, and is now used as the input to
`scan_universe_diagnostics` in Tabs 3, 4, and 5 specifically (avoids a
second network round-trip; the theta-mechanism backtests in those same
tabs still receive the FULL longer-window `prices_df`, unaffected). **Tab 2
deliberately left untouched**, per explicit instruction carried over from
Etap 7j/7k ("drugiej nie ruszaj niech bedzie jaka jest") -- it already
screens on 5 years and was never part of this bug. Verified: gate counts
now agree exactly between a fresh 5-year fetch and a 5-year slice taken
from a 10-year fetch, for the same underlying data.

**Second issue found in the same pass:** the "Zwrot Skumulowany 12M [pp]"
column name and label were hardcoded with "12M" in both `engine/pairs.py`
(the underlying DataFrame column) and the Tab 5 UI, left over from before
`n_months` became adjustable (Etap 7k) -- so the column kept saying "12M"
even when the project owner ran the analysis with, say, 24 or 48 months.
Fixed: the engine column is now generically named "Zwrot Skumulowany [pp]"
(no month count baked in), and Tab 5's table header is built dynamically
as `f"Zwrot Skum. {n_months}M [pp]"`, reflecting whatever value was
actually used for that run.

**Third issue, a genuine pre-existing bug caught incidentally while
verifying the above changes end to end:** `compute_correlations` broke
silently (numpy `RuntimeWarning`s about invalid subtraction/dot-product,
originating inside `pandas.Series.corr()`) whenever a batch legitimately
contained a pair with `Half-Life = float("inf")` (a documented, valid
output of `_half_life` for a spread showing no mean reversion at all --
not a rare edge case, several real universe pairs hit this). `+-inf`
is NOT handled the way NaN is by the underlying correlation computation
(NaN is excluded pairwise; inf corrupts the whole calculation). Fixed:
both the candidate column and the target column are now passed through
`.replace([np.inf, -np.inf], np.nan)` before correlating, letting
pandas's existing NaN-pairwise-exclusion handle it correctly. Verified
with a direct test (a 5-row DataFrame with one `inf` Half-Life value):
before the fix, real `RuntimeWarning`s were raised; after, zero warnings,
and the correlation for that column was correctly computed from the
remaining 4 valid rows rather than being corrupted by the 5th.

**New clarifying chart in Tab 5** (addressing "nie wiem na jakiej podstawie
[Zwrot Skumulowany] są liczone... wyciągnięte z dupy"): a scatter of
"Śr. Alpha Miesięczna" (x) vs. "Zwrot Skumulowany" (y) for the current
batch, with a dashed reference line showing what the naive (non-compounded)
`mean * n_months` product would look like -- makes visible, directly on
the chart, that these are two different SUMMARIES of the same underlying
per-month results (closely related, not independent or arbitrary), and
that true compounding diverges from the naive product because it also
depends on the ORDER of returns, not just their average.

**Also added, addressing "jak ty pozycjonujesz w rankingu... nadal tego nie
rozumiem":** explicit UI copy directly above the ranking table in Tab 5,
stating plainly that rows are sorted by `|t-statystyka|` descending (not
by raw Alpha), that this measures consistency relative to a pair's own
month-to-month variability rather than sheer average size, and that the
table's sort order has NO effect on the correlation bars or scatter
charts above it (correlations are computed from the full unsorted set of
pairs regardless of on-screen ordering).

Full app integration recheck (57/57 callbacks -- same count, this pass
modified existing callback bodies rather than registering new ones),
solver regression rerun (identical weights to every prior check in this
project's history).

---

### 7m. Rejected Experiment — Whole-Period Daily-Averaged Divergence ("Pomysł 1")

Confirmed hypothesis tested (2026-09-08, thirteenth follow-up), and
confirmed REJECTED after empirical testing -- recorded here specifically
so this is not attempted again without knowing it was already tried and
found not to work. Motivation: Tab 5's per-pair t-statistics (based on
12-48 discrete monthly snapshots) looked noisy/inconsistent; hypothesized
that averaging a SIGNED log-divergence between the theta-tilted equity
curve and the 50/50 benchmark curve at DAILY resolution across the WHOLE
multi-year test window (1000+ observations instead of 12-48) would give a
materially less noisy per-pair estimate.

**Built** (`mean_signed_log_divergence`, `run_theta_divergence_batch` --
kept in `engine/pairs.py` as a working, tested function, but NOT wired
into any UI tab): computes `mean_t(ln(equity_strategy(t)) -
ln(equity_benchmark(t)))` across every day of the test window.

**Rejected, confirmed by direct measurement, not assumption**: comparing
this metric's coefficient of variation across 15 repeated random draws of
the same underlying generative process against the existing monthly
mean_alpha's own CV showed the NEW metric was, if anything, slightly
WORSE (CV 0.67 vs 0.52), not better. Root cause, confirmed directly:
lag-1 autocorrelation of the daily divergence LEVEL series was 0.9924, and
even lag-21 (month-to-month) autocorrelation was 0.8605 -- both
essentially describe the SAME cumulative path, not independent
observations. Since the theta mechanism only makes a NEW decision once
per `rebalance_days` (weight is frozen for the whole period), the true
number of independent "bets" the mechanism makes is bounded by
`n_months`, regardless of how finely the resulting cumulative curve is
sampled -- averaging at daily resolution over a *cumulative* curve does
not manufacture new independent information, it just re-weights the
estimate toward however much cumulative drift has accrued by late in the
window. This is a genuinely useful negative result, arrived at by testing
the actual hypothesis rather than assuming it, and is why the module's
real solutions for more statistical power remain Etap 7k's Option A (more
independent months per pair) and Option B (pooling across pairs) --
neither of which tries to extract more information from a single
already-fixed set of monthly decisions.

### 7n. Within-Period Measurement Precision ("Opcja B") and Window-Length Comparison (complete)

Distinguished explicitly from Etap 7m's rejected approach, per the
project owner's own clarification: NOT averaging across the whole
multi-year cumulative curve (which failed), but improving the measurement
of EACH already-independent period individually, by using the mean of
DAILY signed tilt-effects WITHIN that one period instead of just the
period's two endpoint prices. Crucially different because the daily
returns being averaged here are confined to a single already-independent
`rebalance_days`-long window -- they are not points on a persisting
cumulative path spanning multiple decision periods, so this does not hit
the autocorrelation problem that sank Etap 7m.

**Verified BEFORE trusting it**, learning directly from the Etap 7m
mistake: a controlled test held a known, fixed weight tilt and a known
true expected daily divergence constant, then compared the variance of
the OLD two-endpoint measurement against the NEW daily-averaged
measurement across 3000 random noise realizations of the SAME true
effect. Result: the new method's standard deviation was 95.3% LOWER than
the old method's, and its mean was far closer to the true underlying
value (the old method's endpoint-based measurement is itself distorted by
the compounding of the whole price path between the two endpoints, not
just the average daily separation). A follow-up test across 20 random
seeds of the full `simulate_monthly_walkforward` function (not just the
isolated calculation) showed mean |t-statistic| around 1.9 with 11/20
draws crossing conventional significance -- a marked improvement over
prior runs on comparable synthetic data.

**Implementation** (`simulate_monthly_walkforward`): the two-endpoint
`ret_a_month`/`ret_b_month` values are still computed and still used for
TRUE compounding in `cumulative_alpha_pp` (which needs the actual realized
return, not a proxy). Separately, `monthly_alpha_pp` -- the value that
feeds the t-test, mean, and std -- is now computed as
`(weight_a - 0.5) * mean_t(daily_ret_a(t) - daily_ret_b(t))` across every
trading day within that one period, using `pandas.Series.pct_change()`.

**Window-length comparison, confirmed follow-up question** ("czy nie
lepiej wykorzystać np. 2 miesięczne okna i 24 okresy, lub 3 miesięczne i
16 okresów"): a genuine bias-variance tradeoff -- fewer, LONGER
independent periods average out more within-period noise per observation,
but leave fewer independent observations overall; MORE, SHORTER periods
give more observations but each is a noisier single decision. Not assumed
to favor either side -- answered empirically per-universe. New function
`compare_window_lengths` runs `run_monthly_walkforward_batch` three times
over the SAME total ~4-year span and the SAME candidate pairs
(`WINDOW_LENGTH_VARIANTS`: 1mo×48, 2mo×24, 3mo×16), all using the same
Etap 7n precision fix for a fair comparison, and returns both a per-pair
detail table and a per-variant summary (mean |t-statistic|, % of pairs
individually significant). On one synthetic test batch (fast, well-scaled
mean reversion, half-life ~10 sessions), 1mo×48 clearly outperformed the
longer variants (mean |t|=4.16, 100% significant, vs. 2.85/83% and
2.20/67%) -- consistent with the intuition that periods much longer than
a pair's own reversion half-life let multiple reversion cycles wash out
within one measurement, but this is one synthetic scenario, not a general
rule; the real answer depends on the actual universe's pairs and is what
this new tab is for.

**UI**: sixth `dcc.Tab` ("PORÓWNANIE DŁUGOŚCI OKNA") -- minimum-gates
filter, theta input, a grouped bar+line chart (mean |t-statistic| as
bars, % significant as an overlaid line) comparing the three variants at
a glance, and a detail table with every (pair, variant) combination.
Self-contained fetch (10y, screened via the Etap 7l `_screening_slice`
fix for consistency with the rest of the module).

Full app integration recheck (58/58 callbacks -- 57 prior + 1 new), solver
regression rerun (identical weights to every prior check in this
project's history).

---

### 7o. Pair Assignment via Maximum Weight Matching — informational Tab 1 panel (complete)

Confirmed follow-up (2026-09-08, fifteenth follow-up): with the persistence
testing methodology (Etap 7n's within-period precision fix, the three
window-length variants) now producing sensible results on manual review,
the project owner raised the natural next problem: many companies qualify
for MULTIPLE persistent pairs simultaneously (e.g. A pairs well with both
B and C), and picking greedily ("take the best pair, then the next")
can strictly underperform a globally optimal assignment. Confirmed as
exactly the well-known Maximum Weight Matching graph problem, with an
EXACT polynomial-time solution (Edmonds' Blossom algorithm) rather than
a heuristic to invent.

**Persistence threshold, confirmed after the project owner's own review of
several dozen pairs**: raised from an initially-proposed fixed t-statistic
cutoff (1.660) to a ONE-SIDED p-value threshold of 0.10, combined with a
hard requirement that t-statistic > 0. Two reasons, both confirmed
directly: (1) the critical t-value for "p<0.05" depends on degrees of
freedom, which differs across the three window-length variants (df=47 for
1mo×48, df=23 for 2mo×24, df=15 for 3mo×16) -- a single fixed t-statistic
threshold is not equivalent across variants, while p-value is
automatically the correctly-scaled version of the same information; (2)
`scipy.stats.ttest_1samp`'s p-value is two-sided by construction, but only
POSITIVE persistence should ever qualify a pair -- confirmed directly via
the project owner's own observed counter-example (ASML/000660.KS,
persistently WORSE than 50/50) that a plain two-sided p-value threshold
would wrongly admit a significantly NEGATIVE pair. Fixed by halving the
two-sided p-value ONLY when t-statistic > 0 (defaulting to 1.0 — never
qualifying — otherwise), verified directly against this exact reported
pair's numbers (t=-2.463, two-sided p=0.0217) confirming it is correctly
excluded despite what would otherwise look like a "significant" p-value.

**Confirmed candidate-pool scope**: the persistence test runs against the
FULL C(n,2) combination set among whichever tickers are CURRENTLY CHECKED
in Tab 1's universe checklist (Etap 5) -- NOT gate-prefiltered at all (no
`min_gates` requirement here), since this Tab 1 panel already operates on
a small, pre-selected rebalance-candidate set (~30-40 names), unlike the
Relative Value module's separate large-universe scanner. The project
owner separately confirmed (same follow-up) that even the loose
`min_gates=1` prefilter used elsewhere in the Relative Value module should
NOT be a precondition for THIS mechanism: since formal cointegration gates
have shown weak-to-negative correlation with actual persistence
throughout this whole project arc, requiring them as a gate here would
reintroduce exactly the bias this new methodology exists to correct
(confirmed real-world motivating case: memory-sector pairs with weak
formal p-values but genuine theta-mechanism persistence).

**New engine functions**:
- `compute_persistence_qualifying_pairs`: runs `compare_window_lengths`
  (Etap 7n) for every candidate pair, applies the one-sided p<0.10 & t>0
  filter with "OR" logic across the three variants (a pair needs to
  qualify in only ONE window length, not all three), and reports the
  BEST (highest) t-statistic among the variants that individually
  qualified as the edge weight -- confirmed weight choice, since
  t-statistic combines both the size and the month-to-month consistency
  of the edge, rather than average alpha alone (which could reward a
  single lucky period).
- `run_maximum_weight_pair_matching`: builds a `networkx.Graph` (nodes =
  companies, edges = qualifying pairs weighted by their best t-statistic)
  and runs `networkx.max_weight_matching` (exact Blossom algorithm).
  Verified directly against the textbook counter-example that motivates
  using an exact algorithm over a greedy one (edges A-B=10, A-C=9, B-D=9
  -- greedy picks A-B alone for total weight 10; the true optimum is
  A-C + B-D for total weight 18) before trusting the library call for
  this project; confirmed the full pipeline (persistence filter →
  matching) preserves this property end to end on synthetic
  theta-mechanism-generated data, with every company appearing in AT MOST
  one selected pair by construction.

**UI**: new informational panel in Tab 1 ("RELATIVE VALUE -- DOBÓR W PARY"),
appearing automatically once Stage 1 ingestion populates `store-raw-close`
(a small, standalone visibility-toggle callback was used instead of adding
another Output to the large, already-repeatedly-regression-tested
`run_stage_01_ingestion` function, to keep this addition at zero risk to
it). Two complementary visualizations, confirmed jointly by the project
owner after reviewing an inline mockup first:
- **Network graph** (circular/deterministic layout, not a force-directed
  one -- chosen for predictability at any node count rather than
  aesthetics that can vary unpredictably by graph structure): thin dotted
  gray edges for every qualifying-but-unselected pair, thick green edges
  for the selected matching, green-filled nodes for matched companies,
  dim gray nodes for unmatched ones.
- **Matrix heatmap**: full N x N grid of every currently-selected ticker
  that has at least one qualifying connection, cell color = t-statistic
  (sequential colorscale, blank/NaN where no qualifying pair exists),
  with a green border overlay (`fig.add_shape`) specifically marking the
  cells corresponding to the SELECTED matching -- confirmed as the primary
  tool for larger universes (30-40+ names), where the network graph's
  crossing lines would become hard to read; the project owner explicitly
  noted a circular network layout would likely become cluttered at that
  scale, motivating the matrix as a complementary, better-scaling view
  rather than a replacement.
- A detail table lists every selected pair plus every unmatched ticker,
  distinguishing "never had any qualifying pair" from "had a qualifying
  pair but lost the competition for the final matching" (both fall back
  to the existing Ward/DTW/RMT clustering).

Verified end to end on a controlled 8-ticker synthetic scenario (3
deliberately-constructed genuine pairs plus 2 unrelated tickers): the
matching correctly identified and selected all 3 designed pairs; the 2
unrelated tickers were correctly reported as unmatched, with the
diagnostic message correctly attributing this to "lost the competition"
rather than "no qualifying pair at all" once it was confirmed (a real,
informative side-effect of the p<0.10 threshold across a moderate number
of combinations) that a few additional spurious near-qualifying
connections had, in fact, formed by chance among the unrelated tickers --
exactly the kind of multiple-comparisons effect flagged as a risk earlier
in this project's discussion of large-scale brute-force pair search
("Pomysł 2"), observed here directly even at a small, controlled scale.

Full app integration recheck (60/60 callbacks -- 58 prior + 2 new), solver
regression rerun (identical weights to every prior check in this
project's history). `networkx>=3.2` added to `requirements.txt`.

**Explicitly still deferred, per the project owner's own confirmed
sequencing**: this pair-assignment mechanism remains PURELY INFORMATIONAL
-- it does not yet feed into Stage 1's SLSQP weights, and the
uncertainty-in-denominator TPS reformulation discussed the same follow-up
is a separate, not-yet-implemented step.

---

### 7p. Pair-Order Canonicalization + Unified Qualification Criterion — IN PROGRESS (partial, confirmed scope)

Confirmed report (2026-09-08, sixteenth follow-up): manually re-checking
pairs from the Etap 7o matching panel found that swapping which ticker is
"A" vs "B" for the SAME pair (e.g. MCHP/SMTC) gave meaningfully different
diagnostics and backtest results. Root cause, confirmed directly, not
guessed: OLS regression is NOT symmetric -- `log(A) ~ log(B)` fits a
genuinely different line than `log(B) ~ log(A)` unless correlation is
perfect -- so hedge_ratio, spread, Z-score, and every theta-mechanism
result downstream differ by direction. This was always true of
`_hedge_ratio_and_spread` and `test_pair_cointegration`/`coint()`, but had
never been explicitly flagged before this follow-up. Directly connected,
same follow-up: a separately-reported chart pattern where the "worse"
direction's weight almost never crosses 50/50 -- confirmed explanation: a
poorly-fitting regression direction leaves a residual/spread that is more
persistent drift than genuine mean-reversion, so the Z-score (and
therefore weight) gets stuck on one sign for long stretches against a
252-session ROLLING baseline, rather than oscillating as a true
mean-reverting spread would.

**Second, related confusion resolved the same follow-up**: the project
owner found pairs from the Etap 7o matching panel (qualified there) that
"failed" cointegration p-value when checked manually, and asked to unify
this. Root cause: two DIFFERENT tests were both being labeled "p-value" --
the formal Engle-Granger cointegration p-value (tests whether price
LEVELS are linked) vs. the theta-mechanism persistence test's p-value
(one-sample t-test on 12-48 monthly alpha values, tests whether the
mechanism's own realized edge differs from zero). A pair can fail
cointegration in both directions while still showing genuine, persistent
theta-mechanism outperformance -- not a bug, but a naming collision that
made two unrelated numbers look contradictory. **Confirmed resolution**:
theta persistence (one-sided p<0.10, t>0, Etap 7o's own criterion) becomes
the SOLE qualification criterion EVERYWHERE in this module going forward;
cointegration becomes PURELY INFORMATIONAL, never a gate, clearly labeled
as such wherever it still appears. Also confirmed and worth recording
precisely: the project owner's own restatement of what the theta
persistence p-value measures -- "ocena różnicy dynamicznego portfela od
portfela 50/50, tylko nie uwzględnia w którą stronę, dlatego t-statystyka
musi być dodatnia" -- is exactly correct and was confirmed as such: the
two-sided t-test p-value alone cannot distinguish a significant POSITIVE
edge from a significant NEGATIVE one (a persistently-losing pair scores
just as "significant" on p-value alone), which is precisely why t>0 is a
separate, mandatory second condition, not a redundant one.

**New engine function**: `canonicalize_pair_order_by_theta(prices_x,
prices_y, ticker_x, ticker_y, theta, train_years)` -- runs ONE
representative theta-mechanism backtest (1mo x48, the same reference
granularity already used as compute_persistence_qualifying_pairs's own
sort key) in EACH direction and returns whichever assignment has the
higher t-statistic as canonical. Confirmed criterion (2026-09-08,
seventeenth follow-up, explicit): the SAME theta-persistence criterion
used for qualification everywhere else, NOT cointegration p-value --
self-consistent, since a direction that already fails to qualify at all
can never win the canonicalization either. Deliberately uses a single
1x48 run rather than the full three-variant `compare_window_lengths` in
both directions, to keep the added cost to one extra
`simulate_monthly_walkforward` call per candidate pair rather than six.
Verified directly against a constructed asymmetric-noise scenario (one
ticker with small idiosyncratic variance, the other with materially
larger added noise on top of the same true mean-reverting spread --
exactly the kind of heteroscedasticity that produces OLS direction-
asymmetry): the two directions gave t=3.530 vs t=3.142, and
canonicalization correctly selected the higher one. Falls back to the
original, unchanged order (with t_stat_used=0.0) when neither direction
produces a usable result, rather than raising.

**A second, independently-discovered bug, found while wiring
canonicalization into the Etap 7o matching panel**: that panel reused
Stage 1's `store-raw-close`, which is deliberately fetched at only 5 years
(fine for clustering, Stage 1's own purpose) -- but the theta-persistence
mechanism needs up to ~9-10 years for its 48/24/16-month variants to have
genuine 5-year training windows throughout. With only 5 years available,
early test months were silently skipped entirely, and even the LATER,
"usable" months were computed against a shortened, non-ideal training
window -- with no warning this compromise was happening. This plausibly
degraded the panel's real-world results without ever surfacing an error.
**Fixed**: the panel now performs its OWN independent 10-year fetch
(reading `store-raw-close` only to learn which tickers are currently
selected, not for their price values) -- Stage 1's own 5-year fetch for
clustering is completely untouched.

**Confirmed and applied so far** (2 of the identified 3+ entry points):
1. Tab 1's manual pair-picker (`analyze_pair`): now canonicalizes order
   before computing diagnostics, and now computes + displays the theta-
   persistence qualification check (✓/✗, with its own t-statistic) as the
   PRIMARY result, with every cointegration-derived badge explicitly
   relabeled "(informacyjnie)" and the old "gates_passed/3 -- kwalifikuje
   się" summary line removed (cointegration no longer gates anything, so a
   pass/fail summary based on it was actively misleading to keep). Verified
   directly: calling this function with the same two tickers in BOTH
   orders now produces byte-identical `hedge_ratio` and canonical
   ticker_a/ticker_b in both cases.
2. Etap 7o's Rebalance-tab matching panel: canonicalizes every candidate
   combination before building its pairs list, on top of the independent-
   fetch fix above.

**Explicitly NOT yet done, scope confirmed but deferred to a following
turn**:
- `scan_universe_diagnostics` itself (the underlying universe-wide
  cointegration scanner, feeding Tabs 1's own scan button, 2, 3, 4, 5, 6's
  candidate-pair lists) does not yet canonicalize its own (Ticker A,
  Ticker B) assignment, which today comes from whatever order
  `itertools.combinations` happens to produce over the universe ticker
  list. An open question was posed to the project owner and not yet
  answered as of this entry: since cointegration is now purely
  informational everywhere, does the SAME pair showing different
  informational cointegration numbers in different tabs (depending on
  which of several inconsistent fetch windows was used to screen it --
  5y in Tabs 1/2, 10y-sliced-to-5y via `_screening_slice` in Tabs 3/4/5/6)
  still need to be unified, or is this now acceptable since nothing gates
  on it any more.
- The broader UI relabeling pass (distinguishing "p-value (kointegracja,
  informacyjnie)" from "p-value (trwałość theta)" consistently across
  Tabs 2, 3, 4, 5, 6, and reconsidering whether those tabs' `min_gates`
  cointegration-based dropdown filters should be replaced by a
  theta-persistence-based filter instead, for full consistency with the
  new confirmed single-criterion design) has not yet been started.

Full app integration recheck after this partial pass (60/60 callbacks --
same count, modified existing callbacks), solver regression rerun
(identical weights to every prior check in this project's history).

---

### 7q. Scanner Canonicalization + Live Progress Reporting for Long-Running Callbacks (partial, confirmed scope)

Confirmed follow-up (2026-09-08, seventeenth follow-up), continuing Etap
7p. Two confirmed decisions from the project owner before implementing:
(1) canonicalize `scan_universe_diagnostics` too, accepting that this
changes DISPLAYED cointegration numbers (P-Value/Half-Life/Hedge Ratio)
for pairs where direction matters -- corrected the project owner's own
initial assumption that this would be purely cosmetic (column display
order only); (2) Tabs 2 and 3 (the discrete 80/20-mechanism tabs) keep
their own cointegration gate unchanged -- unification to the theta-
persistence criterion applies only where the theta mechanism itself is
being screened or evaluated (the scanner, Tabs 4/5/6, the Rebalance-tab
matching panel).

**Scanner canonicalization**: `scan_universe_diagnostics` gained an
OPTIONAL `full_history_df` parameter. When supplied (a longer, ~10-year
price history for the same tickers), each pair is run through
`canonicalize_pair_order_by_theta` BEFORE its cointegration diagnostics
are computed, using the SAME theta-persistence criterion as everywhere
else in this module (not cointegration itself -- confirmed 2026-09-08,
same follow-up: order choice should track the criterion that actually
matters, not the one being demoted to purely informational). When omitted
(the default), behavior is unchanged -- used by Tab 1's own "Skanuj
Uniwersum" button and Tab 2, neither of which fetch enough history (only
5 years) for the theta-based check regardless, and neither of which is
being altered by this follow-up. Wired into the 3 callers that already
fetch 10 years (Tabs 4, 5, 6's `_screening_slice`-based candidate
selection) -- Tab 3's own `scan_universe_diagnostics` call (6-year fetch,
discrete mechanism) deliberately left untouched.

Verified directly with a constructed scenario where the natural
`itertools.combinations` order (driven by dict/column insertion order)
disagreed with the theta-optimal direction: without `full_history_df`,
the scan reported (NOISY, CLEAN) with P-Value=1.01e-07, Half-Life=9.9;
with it supplied, the scan correctly flipped to (CLEAN, NOISY) with
P-Value=2.96e-07, Half-Life=10.6 -- concretely demonstrating both that
the fix changes real displayed numbers (not just column layout) and that
it correctly tracks the higher-t-statistic direction.

**Second, independently-raised problem, same follow-up**: running
several Tab 4/5/6 analyses back-to-back took roughly 15 minutes with only
a spinner and zero progress feedback -- confirmed as a real, separate
usability problem worth fixing regardless of the min_gates/theta-gate
question still pending (see below). **Infrastructure built**: Dash's
`background=True` callback mechanism, backed by `DiskcacheManager`
(`ui/app_instance.py` -- `diskcache` and `multiprocess` added to
`requirements.txt`; the latter is `DiskcacheManager`'s own runtime
dependency, not obvious from the package name, discovered only when the
app failed to import after adding the manager). Confirmed available in
Dash 3.x (predates the 4.x rewrite the project is deliberately avoiding),
so this does not conflict with the `dash<4.0` pin.

**Engine-level plumbing, confirmed additive and UI-agnostic**: an optional
`on_progress` callback parameter was threaded through the THREE layers a
Tab 5 run actually calls -- `run_monthly_walkforward_batch` (innermost,
calls `on_progress(i, total)` after each pair) -> `compare_window_lengths`
(wraps it per-variant, calling `on_progress(variant_label, i, total)`) ->
`compute_persistence_qualifying_pairs` (passes it straight through). None
of these engine functions import Dash or know anything about callbacks --
existing callers that don't pass `on_progress` see no behavior change,
verified directly (a 4-pair test confirmed exactly `[(1,4),(2,4),(3,4),(4,4)]`
progress calls; a 3-pair/3-variant test through all three layers confirmed
exactly 9 calls in the correct variant-labeled order).

**Wired into Tab 5 (`run_theta_batch_attribution`) as the first
end-to-end proof of concept**: `background=True`, a `progress=[Output(
"relval-thetabatch-status", "children", allow_duplicate=True)]` reusing
the SAME status element as the callback's own final Output (confirmed
this dual role is valid in Dash -- interim `set_progress` updates show
live text, the function's own return value becomes the final message once
it completes; `allow_duplicate=True` was required and resolved a
duplicate-output registration error caught immediately when the app tried
to import). `set_progress` is now the callback's first positional
argument (Dash's convention when `progress` is specified). Verified
end-to-end on a synthetic 8-ticker/22-qualifying-pair scenario by calling
the function directly with a fake `set_progress` collector: 27 progress
messages were generated, in the correct order, from
"Wczytuję listę spółek..." through per-pair "para 1/22" ... "para 22/22"
to the final rendering step -- concretely confirming what the project
owner will actually see change on screen during a long run, rather than
the button appearing to hang.

**Explicitly NOT yet done, confirmed scope for a following turn**:
- The SAME `background=True` + progress-reporting treatment has not yet
  been applied to Tab 4 (`run_monthly_persistence_test`), Tab 6
  (`run_window_length_comparison`), or the Rebalance-tab pair-matching
  panel (`run_pair_matching_analysis`) -- all three call the same
  progress-capable engine functions, so extending the pattern should be
  mechanical, but has not been done or tested yet.
- The confirmed two-stage performance compromise for replacing Tabs
  4/5/6's cointegration-gate min_gates filter with a theta-persistence
  gate (cheap single-variant theta screen first, full three-variant test
  only on pairs that pass it -- confirmed 2026-09-08, same follow-up, to
  avoid paying the full theta-backtest cost on every possible combination
  in a large universe with no cheap pre-filter at all) has NOT been
  implemented yet. Tabs 4/5/6 still filter candidate pairs by cointegration
  `Gates Passed >= min_gates` as before this follow-up.
- A separately-raised, explicitly deferred calibration idea (not yet
  designed or scheduled): today's Maximum Weight Matching algorithm
  (Etap 7o) will always prefer using MORE pairs over fewer, since every
  qualifying edge has positive weight and adding one never decreases the
  total -- the project owner correctly noted this can force a company into
  a marginal pairing it would be better off leaving to clustering instead,
  and wants a separate quality threshold for "is this pairing worth pulling
  companies out of clustering for" on top of the existing bare qualification
  test -- explicitly filed as a future calibration step, not started.

Full app integration recheck after this partial pass (60/60 callbacks --
same count so far), solver regression rerun (identical weights to every
prior check in this project's history).

---

### 7r. Reloader Fix, Two-Stage Theta Gate Rollout, Progress Reporting Everywhere (complete)

Confirmed follow-up (2026-09-08, still eighteenth follow-up) completing
Etap 7q: the project owner reported running Tabs 1, 5, 6 and seeing NO
progress indication anywhere, including Tab 5 where Etap 7q's proof of
concept was supposedly already wired and tested.

**Root cause, found and fixed**: `app.py` ran with `app.run(debug=True,
port=8050)`. `debug=True` enables Werkzeug's auto-reloader by default,
which spawns a SEPARATE monitor process on top of the actual app process
-- a documented source of `DiskcacheManager` failures, since the
background-callback multiprocessing worker can end up talking to the
wrong process, so `set_progress` updates (and sometimes the whole
background task) never reach the browser, with no error surfaced
anywhere. Fixed: `app.run(debug=True, use_reloader=False, port=8050)`
-- `debug=True` itself (error pages, callback exception detail) is kept;
only the reloader is disabled. The person restarts the server manually
after code edits while this is off, since it no longer auto-restarts.

**Two-stage theta gate, completed and rolled out to all four places
identified in Etap 7q as still pending**:
- New engine function `compute_theta_qualifying_pairs_two_stage`
  (Stage 1: cheap canonicalization-based screen via
  `canonicalize_pair_order_by_theta`, keeping only pairs whose
  best-direction t-statistic exceeds `CHEAP_SCREEN_T_THRESHOLD` (0.0) --
  Stage 2: the full three-variant `compute_persistence_qualifying_pairs`
  check, run only on Stage-1 survivors).
- Verified this two-stage result EXACTLY matches a fair, non-shortcut
  comparison (canonicalize every possible combination first, then run
  the full three-variant check on all of them with no cheap-screen
  exclusion) on a 7-ticker test universe: 34 qualifying pairs in both
  cases, identical sets -- confirming the cheap screen did not wrongly
  exclude any true qualifier on this test. (An earlier, naive comparison
  attempt without first canonicalizing the "full" side gave a false
  mismatch -- caught and corrected before trusting the result, since that
  comparison wasn't apples-to-apples.)
- Stage 1 was further extracted into its own standalone
  `cheap_theta_screen_candidates` function, confirmed necessary
  specifically for Tab 6: that tab's own purpose IS running the full
  three-variant `compare_window_lengths` comparison, so calling the
  combined two-stage function there would redundantly re-run it a second
  time. Tab 6 now uses the cheap screen alone as its sole candidate
  pre-filter -- its own `min_gates` dropdown was retired from filtering
  entirely for this tab specifically (confirmed reasoning: pre-filtering
  by "qualifies in >= N windows" before the comparison that answers
  exactly that question would hide the very near-miss cases -- pairs
  qualifying in only 1 or 2 of 3 windows -- that the tab exists to show).
  Tabs 4 and 5, and the Rebalance-tab matching panel, use the combined
  two-stage function directly and keep their `min_gates`-equivalent
  dropdown, now filtering on "Liczba okien qualif." (how many of the 3
  window variants each pair qualified in) instead of cointegration gate
  count -- same UI control, same 1/2/3 options, new underlying meaning,
  confirmed and applied consistently.
- The Rebalance-tab matching panel (Etap 7o/7p) had NOT been using the
  cheap screen at all -- it manually canonicalized every possible
  combination and then ran the full three-variant check on ALL of them
  unconditionally, exactly the expensive case the two-stage compromise
  was meant to eliminate. Fixed to call
  `compute_theta_qualifying_pairs_two_stage` directly.
- Cointegration is carried through as a merged, clearly-labeled
  informational overlay in Tab 5's table ("(informacyjnie)" suffix on
  every cointegration-derived column header) rather than dropped
  entirely -- confirmed still useful to see, just never as a gate.

**Progress reporting extended to match**, using the exact pattern proven
in Etap 7q's Tab 5 proof of concept (`background=True`, a `progress`
Output reusing the tab's own status element with `allow_duplicate=True`,
`set_progress` as the callback's first argument): applied to Tab 4
(`run_monthly_persistence_test`), Tab 6 (`run_window_length_comparison`),
and the Rebalance-tab matching panel (`run_pair_matching_analysis`).
Coarse stage-level messages ("Liczę IS Score...", "Liczę Trailing
Score...", "Uruchamiam optymalne skojarzenie...") are used around
`run_walk_forward_validation` and `run_theta_trailing_stability` (neither
has per-pair progress plumbing yet, unlike the theta/monthly functions
threaded in Etap 7q) -- fine-grained per-pair/per-variant progress is
shown for every stage that already supports `on_progress`.

Verified end to end on all four callbacks via direct function calls with
a fake `set_progress` collector: Tab 4 produced 153 ordered progress
messages, Tab 5 (retested after its gate replacement) produced 125, Tab 6
produced 151, and the Rebalance matching panel produced 78 -- each
confirmed to start with data-fetch/screening messages, move through
per-pair/per-variant counters, and end at rendering, with the final
message in each case matching what the callback's own return value
reports.

Full app integration recheck (60/60 callbacks -- same count, all changes
modified existing callback bodies/decorators), solver regression rerun
(identical weights to every prior check in this project's history).

---

### 7s. Module-Wide Audit — Unified Naming, Scanner Canonicalization, Stale Text Cleanup (complete)

Confirmed follow-up (2026-09-08, nineteenth follow-up): the project owner
reported the Tab 1 discrepancy was STILL happening despite Etap 7p's
canonicalization fix, and asked for a full module-wide audit unifying
"p-value" so it never means two different things, skipping Tabs 2/3 as
before.

**Root cause of the persisting Tab 1 discrepancy, found and fixed**: the
canonicalization fix from Etap 7p was applied to the MANUAL pair picker
(`analyze_pair`) but never to Tab 1's OWN "Skanuj Uniwersum" scanner
button (`run_relative_value_scan`), which still fetched only 5 years and
never passed `full_history_df` -- so the SAME pair could show one
canonical order in the scanner and a DIFFERENT one in the manual picker,
both within Tab 1. Fixed: the scanner now fetches 10 years and
canonicalizes via the same theta-persistence criterion as everywhere
else. Confirmed the gate itself is UNCHANGED (still cointegration
"Gates Passed", since Tabs 2/3's own candidate selection depends on this
scanner's output and neither is touched by this follow-up) -- only the
(Ticker A, Ticker B) ORDER and therefore the displayed diagnostic numbers
are now consistent. Verified end to end with a controlled test pair: the
scanner, the manual picker, and Tab 5's batch all independently reported
the exact same canonical order (B0/A0) and matching numbers for the same
underlying pair.

**"p-value" disambiguated everywhere it appeared ambiguous**, following
the confirmed rule: cointegration's p-value is always labeled
"(kointegracja)" or "(informacyjnie)"; the theta-persistence mechanism's
p-value (a t-test on monthly alpha values) is always labeled "(t-test)"
or "(trwałość theta)". Fixed: Tab 1 scanner's table column, Tab 5's
pooled-significance panel and its scatter chart's axis title/hover
template/section header, Tab 6's table column. Tabs 2 and 3 explicitly
left untouched, per confirmed scope.

**Two further stale-text issues found and fixed during the audit, unrelated
to p-value but caught in the same pass**:
- Multiple places (Tab 1 scanner's docstring, status message, and layout
  description; Tab 4's layout description) still referenced a "2Y drift
  pre-filter" and "4 bramki" (4 gates) -- both removed from the actual
  mechanism long before this conversation (the drift gate was dropped
  entirely, taking the gate count from 4 to 3), but the TEXT never caught
  up. Fixed to describe the current 3-gate mechanism accurately.
- Several tabs' descriptions (Tab 4, Tab 6, and the scanner's own
  docstring before this follow-up) claimed to "require a prior scan in
  the first tab" -- inaccurate for all of them, since each has always been
  fully self-contained (fetches its own price data independently). Fixed
  to state plainly that each tab is self-contained.

**A vestigial, actively misleading control found and removed**: Tab 6's
"MINIMALNA LICZBA BRAMEK" dropdown no longer did anything after Etap 7r
replaced its candidate-selection logic with the cheap theta screen alone
(confirmed reasoning preserved from Etap 7r: pre-filtering by window-count
before the very comparison that measures window-count would hide the
near-miss cases the tab exists to show) -- the dropdown's value was simply
never read anymore, silently. Rather than leave a control that looks
functional but does nothing, it was removed entirely from the layout AND
from the callback's `State` list and function signature, with the
description text updated to state plainly that candidate selection uses
the cheap screen with no window-count threshold.

**Tab 1 scanner also gained the same progress-reporting treatment**
(`background=True`, `on_progress` threaded into
`scan_universe_diagnostics`'s own loop -- a new capability added to that
function, since it previously had none), since upgrading it to 10 years
with canonicalization roughly doubles its cost per combination, matching
the pattern already proven in Etaps 7q/7r.

Verified end to end: all six affected surfaces (Tab 1 scanner, Tab 1
manual picker, Tab 4, Tab 5, Tab 6, Rebalance-tab matching panel) were
exercised directly via their callback functions with a synthetic
6-ticker/3-pair universe, confirming each runs without error and that the
SAME test pair shows identical canonical order and consistent diagnostic
numbers across the scanner, the manual picker, and Tab 5's batch output.
Full app integration recheck (60/60 callbacks -- same count, all changes
modified existing callback bodies/decorators or removed one dead
State/parameter pair), solver regression rerun (identical weights to
every prior check in this project's history).

---

### 7t. Two-Tier Quality Threshold for Pair Matching (complete)

Confirmed follow-up (2026-09-08, twentieth follow-up), addressing a
calibration gap explicitly flagged and deferred back in Etap 7o/7p: since
`run_maximum_weight_pair_matching` always prefers using MORE edges over
fewer (every qualifying edge has positive weight, so adding one never
decreases the total), it will happily pull a company out of clustering
into a marginal pairing just barely over the qualification bar. Discussed
and confirmed before implementing: qualification (Etap 7o's "does this
pair show ANY persistence") and eligibility-for-matching ("is this pairing
GOOD ENOUGH to be worth pulling companies out of clustering for") are two
different questions and need two different, separately-tunable
thresholds.

**Confirmed two-tier design**: a pair still QUALIFIES under the existing,
looser bar (one-sided p<0.10, t-statistic>0, in at least 1 of 3 window
variants) -- this stays as the criterion for what appears in the
informational scan/table/graph, unchanged. To be ELIGIBLE for the actual
Maximum Weight Matching, a pair must additionally clear a stricter bar,
confirmed jointly with the project owner: qualifying in at least
`MATCH_MIN_WINDOWS` (2) of the 3 window variants -- not just 1 -- since
robustness to how the same total test span happens to be sliced is a
materially stronger signal than one good result from one particular
split; AND a "Best t-statystyka" of at least `MATCH_MIN_T_STATISTIC`
(1.75 -- explicitly chosen by the project owner as a middle ground
between the p<0.05 (~1.645) and p<0.01 (~2.33) reference points offered).

**New engine function** `filter_pairs_eligible_for_matching`, applied as
an explicit, separate step BETWEEN `compute_theta_qualifying_pairs_two_stage`
(or `compute_persistence_qualifying_pairs`) and `run_maximum_weight_pair_matching`
-- the matching function itself is left untouched and general-purpose (it
still just runs the exact Blossom algorithm on whatever graph it's given).
Verified directly with a 4-pair synthetic scenario constructed to trigger
BOTH failure modes independently: a pair with a high t-statistic (1.9) but
only 1 qualifying window was correctly excluded, and a separate pair with
2 qualifying windows but a t-statistic (1.4) below the threshold was also
correctly excluded, while two genuinely strong pairs passed through
unaffected -- confirming the AND condition (not just one or the other)
works as intended, and that neither excluded pair's companies ended up in
the final matching.

**UI**: the Rebalance-tab matching panel now shows THREE tiers instead of
two, both in the network graph (bold green = selected, medium dotted gray
= eligible but lost the matching competition, faint thin dotted = merely
qualifies but below the eligibility bar) and in the detail table (a
distinct "Kwalifikuje się, ale poniżej progu jakości" status alongside the
existing "Wybrana para" / "Dopuszczalna, ale przegrana" statuses) --
confirmed important for transparency: the project owner should be able to
see exactly where the line falls, not just a binary in/out result. The
status message now reports all three counts (qualifying / eligible /
actually selected) instead of two. The panel's description text was
extended to explain both thresholds and why the second one exists at all
(directly referencing the "algorithm always wants one more edge" problem
that motivated this whole follow-up).

Full app integration recheck (60/60 callbacks -- same count, this modifies
an existing callback's body), solver regression rerun (identical weights
to every prior check in this project's history).

**Explicitly still deferred**: whether these two threshold VALUES
(2 windows, t≥1.75) are actually right for the project owner's real
universe is an open empirical question, not something this follow-up can
answer from synthetic data alone -- the project owner is expected to run
this on the real universe and adjust `MATCH_MIN_WINDOWS`/
`MATCH_MIN_T_STATISTIC` if the resulting eligible set looks too strict or
too permissive in practice.

### 7u. Rebalance Tab UI Fixes: Dual Fundamental-Save Buttons, Save-Portfolio in Rebalance, 3D Chart Removal (complete)

Confirmed follow-up (2026-09-18), addressed before starting the post-solver pair-overlay work (Etap 7v+): three independent Rebalance-tab UI gaps found during review.

**Stage 3 fundamental inputs — single confirm button split into two.** Previously, `btn-stage3-confirm` both exported to Stage 4 AND was the only path available at all. Confirmed problem: the project owner sometimes wants to feed the solver deliberately experimental/made-up fundamental values (e.g. stress-testing sensitivity to an extreme analyst spread) without polluting `company_store`'s per-ticker history, which is also read by the Research module. Fixed: "CONFIRM & EXPORT (ZAPISZ DO RESEARCH)" now also calls `company_store.append_entry` for every row; "EKSPERYMENTUJ (BEZ ZAPISU)" performs the identical Stage 4 export with zero persistence calls. Verified directly: save-mode data appears via `get_history()` immediately after, experimental-mode leaves the ticker's history empty.

**Rebalance tab was missing a save-portfolio button — found to already exist, just misplaced.** `save_snapshot_callback` and its full backend (`data/snapshot_store.py`) already existed and already read exactly the Stage-4-relevant stores it needed — but its only button lived in the Sandbox tab's layout, not Rebalance's, so saving a just-computed portfolio required switching tabs. Fixed by extending the SAME callback (not duplicating it) to accept a second button (`btn-save-snapshot-stage4`) placed directly in Rebalance, routed via `callback_context` to the correct one of two status Outputs. Verified directly: both trigger paths tested independently, each updating only its own tab's status element.

**Removed the 3D risk-reward scatter (`render_stage4b_3d_surface`, `graph-stage4b-3d`) entirely** — project owner judged it added no diagnostic value beyond the existing results table + donut chart. Removed the callback function itself, not just hidden, to avoid leaving dead code targeting a nonexistent component.

Full app integration recheck (59/59 callbacks — 60 prior minus the one dead 3D-chart callback removed), solver regression rerun (identical weights to every prior check in this project's history, as expected since all three changes are UI-only with zero `engine/` modifications).

---

### 7v. Post-Solver Relative Value Pair Overlay -- "Droga B" (complete)

Confirmed follow-up: wires the Relative Value pair-matching mechanism (Etap 7o+) into the Rebalance solver's output for the first time, without touching the solver itself.

**Design confirmed jointly**: operates strictly AFTER Stage 1/2 finishes, on the solver's own final weight vector -- never touches mu_i, Sigma_eps, K, or the clustering machinery. This sidesteps the cluster-membership question entirely (a matched pair spanning two different clusters is no longer a problem, since clustering has already finished by the time this runs) and avoids the dead end of nudging mu_i, which is provably impotent for any asset already sitting at the binding w_max ceiling.

New engine functions: `compute_current_pair_tilt` (one-shot "today's" tilt, not a backtest), `apply_post_solver_pair_overlay` (full orchestration: filters to positively-weighted tickers, reuses the existing two-stage theta-persistence gate + Maximum Weight Matching unchanged, redistributes weight ONLY within each matched pair's own combined capital). Confirmed default design choice, explicitly flagged as changeable: ADDITIVE on top of whatever asymmetric split the solver's own fundamentals-driven reasoning already produced between the two legs, not a full replacement with a neutral 50/50 baseline. Every leg clipped to [0, w_max] -- the hard concentration ceiling is never breached, confirmed as a deliberate portfolio-level decision this overlay must not override.

Also produces an `unmatched_with_alternative` diagnostic ("Droga B", chosen over "Droga A" -- letting one stock draw signal from multiple overlapping pairs at once, explicitly rejected as reintroducing Cluster-Throttling/Cascading-Dominance-style unbounded interaction complexity): for any positively-weighted stock that ended up without a pair (lost the 1-to-1 matching competition despite Maximum Weight Matching only optimizing total graph weight, not each node's own outcome), surfaces whether it had another qualifying relationship with a different selected stock -- informational only, no automatic action.

New Rebalance-tab panel, deliberately a SEPARATE, explicit step (own button, `background=True`) rather than wired into the fast, live-reactive main solver callback, since the full persistence gate needs ~10 years of independently-fetched data and would make every slider tweak unacceptably slow otherwise. Purely diagnostic at this stage: does not modify `store-stage4b-results` or what SAVE PORTFOLIO persists.

Verified: pair-sum and total-portfolio-sum invariants hold exactly, w_max never breached even when the solver's own split was already asymmetric, unrelated (unpaired) tickers never touched, the additive blend correctly builds on top of (not replaces) the solver's own within-pair asymmetry, blocking-pair diagnostic correctly surfaces an unused qualifying relationship on a constructed 3-stock scenario. Solver regression unchanged. 60/60 callbacks.

---

### 7w. Pair-Overlay Reproducibility Fix, Z-Score Charts, Sandbox Replication with Live Toggle (complete)

Confirmed bug report: identical-looking runs of the Etap 7v overlay (same theta, same portfolio) produced different results between clicks, and theta=0.5 was observed giving a SMALLER weight change than theta=0.3 -- the opposite of the tilt formula's own linear scaling.

**Root cause, isolated and confirmed**: NOT a math bug. Verified directly on fixed, frozen synthetic data, called repeatedly, that the tilt formula is exactly deterministic and scales exactly linearly in theta (0.5/0.3 ratio reproduced to 4 decimal places). The actual cause: every click of the overlay button re-fetched 10 years of price data over the network from scratch, so two "identical" clicks could silently see different underlying data -- a different current Z-score, or even a different matched-pair set if a t-statistic was hovering near the `MATCH_MIN_T_STATISTIC` threshold. Two runs a few minutes apart were never actually comparing the same inputs.

**Fix -- split into an expensive stage and a cheap stage**, since the two-stage theta-persistence gate is itself theta-invariant (t-statistic doesn't depend on theta's value, confirmed from the original theta-scaling proof) and therefore does not need to re-run on a theta change at all:
- `find_matched_pairs_for_overlay` (expensive): fetch 10Y once, run the gate + matching once, cache the resulting pairs' current Z-scores AND a Z-score history (new `compute_pair_zscore_history`, for charting) -- runs ONLY on an explicit button click.
- `apply_tilts_to_matched_pairs` (cheap): pure arithmetic on the already-cached Z-scores -- safe to call on every theta keystroke, zero network I/O. Verified directly: 4 different theta values against one cached match-set triggered exactly 1 fetch call, not 4.
- `apply_post_solver_pair_overlay` kept as a thin backward-compatible wrapper combining both (existing tests/callers unaffected).

**Z-score charts added** (per matched pair, ~2-year rolling context window, current point highlighted) -- addresses the request to visualize each pair's Z-score, and incidentally makes the "which data was this run against" question visually inspectable going forward.

**Replicated in Sandbox**, with one deliberate difference from Rebalance: a `toggle-sandbox-pair-overlay` checkbox that, once the expensive step has been run at least once, makes the tracked "Manual Sandbox" equity curve ITSELF use the pair-overlay-adjusted weights instead of raw solver weights (legend updates to say "+ RV overlay" when active) -- not just a side diagnostic table, since Sandbox's whole purpose is comparing the tracked impact of different choices. Toggle off, or no cache yet, reproduces prior behavior exactly (verified directly on all three states: off, on-without-cache, on-with-cache).

Shared UI builders (`build_pair_zscore_figure`, `build_pair_overlay_output`) moved to `ui/components.py` so Rebalance and Sandbox render identically without a circular import between the two tab modules.

Verified: full `update_forward_tracker` smoke-tested end to end after substantial surgery to its signature (new Output, 3 new Inputs/State) and body (overlay injection point) -- correct output count, correct new-store population, correct legend/curve behavior across all three toggle/cache states. Solver regression unchanged (NVDA=0.35/HPE=0.246/LYC.AX=0.054/AVGO=0.35). 63/63 callbacks.

---

### 7x. Critical Look-Ahead Bias Fix in Sandbox Pair Overlay, w_max-Ceiling Clarity, Two-Curve Comparison (complete)

Confirmed serious bug report from project owner, reviewing a live screenshot: a matched pair (AMAT/AMD, both legs already at 0.15) showed exactly zero weight change regardless of theta, and separately, the project owner flagged that the Sandbox overlay's data fetch could be seeing prices from AFTER a snapshot's creation date -- a direct violation of this project's own walk-forward principle (Part I Section 6.2, "entirely free of look-ahead and survivorship biases").

**Look-ahead bias, confirmed real and fixed.** `find_sandbox_pair_overlay_matches` fetched `period="10y"` ending TODAY unconditionally, with no awareness of which snapshot it was analyzing or when that snapshot was created. Fixed: the callback now reads `record["created_at"]` and truncates the fetched price history to `prices_df.index <= created_at` before any eligibility test, matching, or Z-score computation runs. Verified directly on a controlled scenario (snapshot dated 30 days ago, 10 years of real data available through today): the resulting Z-score history's last date exactly equals the snapshot's creation date, never later, confirming the pair signal is now only ever informed by data that would genuinely have been available on the day the portfolio was frozen. The Rebalance-tab version of the overlay is unaffected -- it has no historical snapshot involved, so "today" genuinely means today there.

**Zero-effect observation, diagnosed and explained, not silently left as confusing.** Re-verified the tilt math is still exactly deterministic and linear in theta on frozen data (unchanged from Etap 7w). The specific screenshot's zero change was mathematically correct: both legs of the AMAT/AMD pair were already sitting exactly at `w_max`, so `combined = 2*w_max` leaves literally zero room to tilt without breaching the per-leg ceiling -- confirmed by reproducing the exact scenario (`w_max=0.15` gives 0.0000 change, `w_max=0.30` gives a real, non-zero tilt on identical inputs). Added an explicit on-screen note naming this condition whenever both legs of a matched pair sit at the ceiling, instead of an unexplained zero. Also added explicit 4-decimal numeric formatting to the weights table, since the project owner could not tell from the unformatted display whether small real changes were occurring at all.

**New per-pair chart**: weight-share-over-time (`build_pair_weight_history_figure`), derived entirely from the Z-score history already cached for the existing Z-score chart, transformed through the current theta -- zero additional data cost, directly answers "how would this pair's split have evolved over time," which is what the project owner was asking for when comparing this feature to the Relative Value module's own historical weight-evolution view.

**Sandbox comparison redesigned**: the overlay toggle no longer replaces the "Manual Sandbox" curve in place (which silently hid the baseline for comparison) -- it now adds a second, separate "Manual Sandbox + RV overlay" curve alongside the untouched original, so both are visible and directly comparable on one chart. Verified the base curve is provably unmodified when the overlay is added.

**Explicitly deferred, not done in this pass**: the project owner also asked for the full network-graph + heatmap-matrix visualization (matching Tab 1's `run_pair_matching_analysis` panel) inside this overlay. Given the scope of the fixes above was already substantial and safety-critical, this was intentionally left for a following, separate pass rather than rushed alongside a correctness fix -- flagged back to the project owner for confirmation before starting it.

Solver regression unchanged (NVDA=0.35/HPE=0.246/LYC.AX=0.054/AVGO=0.35). 63/63 callbacks.

---

### 7y. Tilt Formula Reversal to Multiplicative-Rescale, w_max Dropped Entirely, Network Graph/Matrix + t-statistics Added (complete)

Confirmed follow-up: the project owner walked through a hand-verified numeric example showing the Etap 7v/7w additive formula was too muted -- a strong pair signal could never meaningfully overpower what the solver had already decided.

**Formula reversal, confirmed with a worked example.** Old: `share_solver + delta_from_theta*tanh(-Z)`, clipped to `[0, w_max]`. New, matched to the project owner's own hand calculation to 4 decimal places (A=0.15, B=0.10, overlay share_A=0.3 -> final A=0.0978, B=0.1522): `share_A_overlay = 0.5+0.5*theta*tanh(-Z)`; `raw_A = weight_A_solver*share_A_overlay`, `raw_B` symmetric; `scale = (weight_A_solver+weight_B_solver)/(raw_A+raw_B)`; `final = raw*scale`. Sanity-verified: at Z=0 this reduces exactly to the original solver weights for any theta -- no signal, no change.

**`w_max` now completely ignored by this overlay** -- explicit, deliberate reversal of the earlier w_max-respecting design, confirmed directly by the project owner ("wmax dla nakładki nie ma znaczenia, ona ma go ignorować i mieć swoją mechanikę"). Verified an extreme Z-score can now push a pair's concentration well past what w_max would have allowed under the old formula -- a real, consciously-accepted tradeoff, flagged clearly rather than silently applied. This also retroactively explains a screenshot the project owner sent showing exactly zero effect on a matched pair (AMAT/AMD): under the OLD formula, both legs were already sitting exactly at w_max, leaving mathematically zero room to tilt without breaching it -- moot now, but confirmed as the correct diagnosis before the formula was replaced.

**t-statistic surfaced end-to-end** -- was being computed (it's literally the Maximum Weight Matching edge weight already) but discarded before reaching the UI. Now captured into `matched_pairs_info` and shown in each pair's chart title and the status line.

**Network graph + t-statistic heatmap matrix added** to the pair-overlay panel in both Rebalance and Sandbox, matching Tab 1's `run_pair_matching_analysis` three-tier visualization (selected/eligible-but-not-selected/qualifies-only) exactly -- extracted as a new shared `build_pair_network_and_matrix` in `ui/components.py`. Required exposing the full qualifying/eligible tiers (not just the final matched set) from `find_matched_pairs_for_overlay`.

Verified: new formula matches the hand calculation to 4 decimal places; neutral-signal sanity check across multiple theta values; total portfolio weight sum still exactly preserved despite `w_max` no longer being enforced; network/matrix section renders correctly on a 5-ticker synthetic scenario (2 pairs matched of 6 qualifying, 4 eligible), with real t-statistics displayed throughout. Solver regression unchanged. 63/63 callbacks.

---

### 7z. Nu (Volatility-Sensitivity) Parameter + Full TPS Formula Breakdown Panel (complete)

Confirmed follow-up, returning to the sensitivity-tooling roadmap deferred while the Relative Value pair overlay was being built: step 1 of that roadmap (add `nu`) plus a new, explicitly requested visibility feature.

**`nu` added**, structurally parallel to the existing `lambda`: `TPS(w) = (w·mu - Rf) / (sigma_p*exp(nu*sigma_p)*exp(lam*k_p) + eps)` -- an additional exponential multiplier on the portfolio's OWN downside-volatility term (`sigma_p`), separate from `lambda`'s multiplier on the crash-overlap term (`k_p`). Threaded through the entire solver chain (`_tps_neg` -> Stage 1 -> Stage 2 -> `run_true_two_stage_optimization` -> `_run_single_pass` -> `run_optimization_with_singleton_split`), default `nu=0.0` everywhere for exact backward compatibility. Confirmed allowed negative (rewarding volatility instead of only penalizing it, per Part I's own thesis that hyper-growth volatility is often upside-skewed alpha). Verified: solver regression unchanged at `nu=0.0`; nonzero `nu` (both signs) measurably moves the allocation on a controlled synthetic test. Wired live into both Tab 4 (auto-generated via `STAGE4A_PARAMS_CONFIG`) and Sandbox (new `slider-sb-nu`).

**Full formula breakdown panel added**, addressing a explicit gap the project owner flagged ("widzę tylko suwaki... chcę widzieć np. czy zmienność wynosi w stosunku do K to jest jakieś 1:8"): a new shared `build_tps_formula_breakdown` (`ui/components.py`) shows every term of the TPS formula with its actually-computed value -- `mu_P`, `Rf`, the numerator, `sigma_P`, `K_P`, the explicitly requested `sigma_P:K_P` ratio, `lambda`, `exp(lambda*K_P)`, `nu`, `exp(nu*sigma_P)`, the full denominator, and `TPS_P` (cross-checked against the solver's own reported value as a consistency guard). Rendered in both Rebalance (`stage4b-formula-breakdown`) and Sandbox (`sandbox-formula-breakdown`), fed from values the solver already returns (`lam_used`/`nu_used`/`rf_used` newly exposed alongside the existing `mu_p`/`delta_p`/`k_penalty_p`/`tps_p`) -- no duplicate computation anywhere.

Verified end to end: Tab 4's panel renders correctly with real values at `nu=2.0` (weights measurably shifted vs the `nu=0` baseline); Sandbox's panel renders correctly at a feasible `w_max`, and gracefully shows a "not yet computed" placeholder rather than crashing when the solver falls back to equal-weight on an infeasible parameter combination (confirmed this is the existing, correct fallback behavior, not a new bug, after a first test run used an infeasible `w_max` for a 2-ticker portfolio by the tester's own mistake). Solver regression unchanged. 63/63 callbacks.

**Explicitly raised but not yet resolved**: the project owner's separate idea of extracting analyst-uncertainty discounting so it no longer affects the ranking value fed to the optimizer while still discounting "expected return" somewhere -- flagged as ambiguous between two structurally different designs (a binary pass/fail quality gate vs. a discount applied to a specific sub-component like `U_adj` only) and, in the gate interpretation, a real risk of worsening the exact NBIX-style adverse-selection failure mode from the earlier rolled-back attempt (Etap: F-matrix-in-denominator) rather than fixing it, since a raw undiscounted ranking value would no longer have any continuous penalty at all once a low bar like `Rf` is cleared. Awaiting the project owner's clarification before any implementation.

---

### 8a. mu_i Gamma Discount Made Universal Across Alpha (resolves the analyst-uncertainty open question from Etap 7z) (complete)

Confirmed follow-up: resolves the open question flagged at the end of Etap 7z. The project owner clarified their idea by describing a concrete symptom observed while sliding `alpha` in the real universe: at low `alpha`, the gamma dispersion discount was so aggressive that as few as ~4 stocks cleared a 25% `Rf` hurdle; at `alpha=1.0`, the discount vanished entirely (by the old design's own stated intent -- gamma was coupled only to the price-target branch, which has zero weight at alpha=1.0), letting through wide-analyst-dispersion, low-forecast-quality names (the project owner's own examples: ARGX, NBIX) purely because nothing was left to discount them.

**Confirmed fix**: `gamma`'s discount (`D_i = exp(-gamma*S_i)`) moved from being coupled to the `(1-alpha)` branch specifically to applying universally to the whole composite return, exactly like `M_i` (revision momentum) and `A_i` (coverage confidence) already do:
```
BaseReturn_i = (1-alpha)*U_raw_i + alpha*G_val_i     # pure weighted average now, no discount inside
mu_i = BaseReturn_i * M_i * A_i * exp(-gamma*S_i)     # D_i now alpha-independent
```
This is an explicit, conscious reversal of the design confirmed back when the Alpha Blend was first introduced (Etap 2) -- at the time, applying gamma at `alpha=1.0` was considered wrong because `S_i` is nominally a price-target-specific dispersion measure with "nothing to discount" once that branch has zero weight. The project owner's own reasoning for reversing this: there is no OTHER available signal specifically measuring the reliability of the EPS-growth estimate itself, so repurposing `S_i` as a general forecast-uncertainty proxy for the WHOLE estimate -- not the price-target branch alone -- is preferable to leaving pure-EPS-growth valuations (`alpha=1.0`) completely undiscounted.

Verified on two constructed example companies (tight-dispersion "AcmeAI" vs wide-dispersion "BetaSemi", same fixed `gamma=1.5`, `kappa=1.0`, `n_ref=8`): `D_i` is now identical (0.558 and 0.120 respectively) across every `alpha` value 0.0/0.3/0.5/1.0 tested for a given company -- confirming the discount is now genuinely alpha-independent. At `alpha=1.0` specifically (where the old formula gave `D_i=1.0`, i.e. no discount at all), BetaSemi's wide dispersion now correctly cuts its 20% raw EPS growth down to 1.10% `mu_i`, against AcmeAI's 25.45% -- closing exactly the asymmetry the project owner identified.

Solver regression: NVDA/AVGO unchanged (still pinned at `w_max` on this fixture), HPE 0.246->0.2463, LYC.AX 0.054->0.0537 -- a small shift on this fixture's specific, moderate `S_i` values (0.13-0.44 across the four test tickers); the dramatic correction is specifically at high-`alpha`/high-`S_i` combinations (like the project owner's real ARGX/NBIX case) that this particular fixture does not exercise. New baseline shown to the project owner; **not yet confirmed** as the new reference value to encode in `CLAUDE.md` as of this entry -- per the project's own standing protocol, the old baseline stays authoritative until explicit confirmation arrives.

`engine/single_asset.py`'s own, structurally different alpha/gamma/kappa/eta model (Research/Company Dossier's per-horizon calibration) is untouched -- confirmed, as before, to be a fully independent model that never calls `compute_composite_upside_row`.

---

### 8b. Global Sensitivity Analysis Infrastructure: Sobol + Optional Morris + PRCC (complete, infrastructure only)

Confirmed follow-up, returning to and completing steps 4-5 of the sensitivity-tooling roadmap in one pass (skipping the originally-planned intermediate single-parameter-sweep/2D-heatmap steps once the project owner confirmed a preference for the full, rigorous method over any cheaper approximation, given real hardware -- i9-14900KF, 32 threads, 64GB RAM -- specifically bought with this kind of workload in mind).

**Extensive design discussion preceded any code**, resolving two foundational questions the project owner correctly flagged as needing answers before anything could be built usefully:

1. **How to combine multiple snapshots.** Confirmed: Sobol analysis operates on ONE frozen snapshot and ONE forward horizon at a time -- never stitches multiple snapshots' data together, since doing so would confound "which parameter caused this" with "which month had different market conditions," and the project owner's own portfolios rotate holdings and fundamental data snapshot to snapshot, making any such mixing uninterpretable. The project owner's own, separate "Połączone Portfolio" idea (plain replacement of one snapshot's tracked curve by the next at its own creation date, no smoothing) is confirmed as a distinct, simpler, independent feature -- not built in this pass, deliberately decoupled from Sobol.

2. **What should the output metric(s) be.** Confirmed CAGR + Sortino + Max Drawdown, all three computed per run rather than picking one -- explicitly to let the project owner compare how a parameter's Sobol indices differ ACROSS the three outputs (a parameter raising CAGR while lowering Sortino or widening drawdown is itself the finding, directly motivated by the project owner's own gamma-at-maximum observation earlier this session -- see the preceding manual-observation commit). Deliberately did NOT use the solver's own TPS_P as a Sobol output, since it would be partially tautological (algebraically built from several of the same parameters being tested). Sortino's MAR is fixed at 0, explicitly NOT tied to the swept `rf` parameter, to avoid mechanically entangling an output's definition with one of the inputs being tested.

**Engine (`engine/sobol_analysis.py`)**: full Saltelli-scheme Sobol (S1 + ST + second-order S2, via SALib), optional Morris elementary-effects screening as a cheap opt-in pre-step (confirmed optional, not forced, per the project owner's explicit preference), and PRCC as a free-riding cross-check computed on the exact same sample (zero additional solver calls). `evaluate_single_parameter_set` is a module-level, picklable function -- the unit dispatched to a `ProcessPoolExecutor` sized to `os.cpu_count()` -- confirmed as the first place in this project where a single task is large enough to actually engage more than one CPU core; this directly answers the project owner's own observation that CPU/RAM/GPU sat unused despite capable hardware, which was correctly diagnosed as "not enough work per task to matter," not a code limitation to remove. Frozen `risk_prices`/`forward_returns` are loaded once per batch and shared read-only across every candidate rather than re-fetched per row.

**Confirmed bug found and fixed during initial testing, before trusting any result**: `w_max`'s sampled range must respect the specific snapshot's own ticker count -- any `w_max` below `1/n_tickers` makes the solver's own equality constraint mechanically infeasible regardless of every other parameter, which is not a real "risk" finding, just content-free wasted compute. Verified this was the sole cause of 100% of a test batch's failures (21/36) before an automatic clamp (`w_max >= 1.05/n_tickers` for the analyzed snapshot) was added; 0/36 after.

**UI**: Sandbox restructured from a flat view into `dcc.Tabs` -- existing content becomes "FORWARD TRACKER", new sibling tab "ANALIZA SOBOLA" added (not a replacement). Snapshot + horizon (1m/2m/3m/6m, all four confirmed) selectors, per-parameter min/max range grid for all 8 parameters (pre-filled from sensible defaults matching the existing UI sliders' own bounds), a live cost estimate (N -> total runs -> estimated seconds on the machine's actual core count, shown BEFORE committing to a run, confirmed requirement), optional Morris checkbox, background-callback execution with live per-batch progress. Results render as grouped S1/ST bar charts with confidence intervals per output, a PRCC table, an optional Morris summary table, and an explicit warning naming the solver-failure rate when nonzero rather than silently smoothing over it.

**Confirmed and tested against the project's actual situation**: a `check_sobol_horizon_availability` guard blocks running a horizon that needs more real trading sessions than have actually elapsed since a snapshot's creation date -- verified directly against the project's real first snapshot (created 2026-09-01, this analysis was built ~18 days later): correctly rejects a 6-month horizon (needs 126 sessions, only ~14 had elapsed) with a clear message rather than silently producing a look-ahead-contaminated or nonsensical result. This is confirmed, explicitly, as PURE INFRASTRUCTURE at this stage -- no historical snapshot yet has enough elapsed real sessions to produce a genuine, trustworthy finding; end-to-end verification in this pass was necessarily done on synthetic data, with real analysis expected to become possible incrementally as real time passes forward from the project's actual first snapshot.

Solver regression unchanged (matches Etap 8a's baseline exactly), as expected since this entire feature is additive and never modifies the solver itself. 66/66 callbacks (+3: horizon-availability check, live cost estimate, main Sobol/Morris/PRCC run).

**Explicitly deferred, not started**: the "Połączone Portfolio" (stitched multi-snapshot tracking) tab remains a separate, simpler, not-yet-built feature, confirmed decoupled from this one.

---

### 8c. "Since Inception" Horizon Added to Sobol Tab (complete)

Confirmed small follow-up, addressing an immediate practical gap found the moment the project owner tried to actually use Etap 8b: the project's own first real snapshot (created 2026-09-01) had only 14 elapsed trading sessions at this point, short of even the smallest fixed horizon (1m = 21 sessions) -- leaving the whole new Sobol tab untestable on real data despite being otherwise ready.

Added `"since_inception"` as an additional horizon option alongside the fixed 1m/2m/3m/6m choices: uses however many real trading sessions have actually elapsed since the snapshot's own creation date, whatever that number happens to be, rather than requiring a fixed count. `check_sobol_horizon_availability` now reports this option as usable once at least 1 session has elapsed (with an explicit noise-warning, not a block, below 5 sessions), and `run_sobol_analysis` sets `horizon_days = available_days` directly for this option rather than a `HORIZON_DAYS` lookup. The existing fixed-horizon behavior and its look-ahead guard are unchanged.

Verified directly against the project's own real scenario (a snapshot with exactly 14 available sessions): "since inception" runs successfully using all 14, while 6m on the same snapshot still correctly refuses (112 sessions short) -- confirming this unblocks exactly the intended case without weakening the existing guard elsewhere. Solver regression unchanged. 66/66 callbacks (no new callbacks).

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

---

**Kontynuacja dziennika w PROJECT_CONTEXT_2.md (od 2026-09-26).** Ten plik
zrobił się bardzo długi, więc chronologiczny, append-only dziennik "Etap X"
kontynuuje się od teraz w `PROJECT_CONTEXT_2.md` (ten sam katalog, te same
zasady: nigdy nie edytować istniejących wpisów, tylko dopisywać nowe na
końcu, pokazać projekt wpisu i czekać na potwierdzenie przed dopisaniem).
Ten plik (`PROJECT_CONTEXT.md`) zostaje nietknięty i zamrożony jako
archiwum -- Etap 1 do Etap 8c włącznie. Numeracja Etapów jest WSPÓLNA
między obydwoma plikami (ciągła, nie zaczyna się od nowa) -- następny
numer po Etap 8c to Etap 8d, w `PROJECT_CONTEXT_2.md`.