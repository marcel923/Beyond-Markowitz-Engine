"""
engine/sobol_analysis.py
==========================
Global sensitivity analysis (Sobol + optional Morris screening + PRCC) for
the Stage 4 TPS solver's 8 parameters (alpha, lambda, nu, gamma, kappa,
w_max, Rf, n_ref), confirmed design 2026-09-19.

CORE DESIGN DECISION (confirmed at length before any code was written --
see PROJECT_CONTEXT.md for the full discussion): this module evaluates
sensitivity for ONE FROZEN, HISTORICAL snapshot at a time, over ONE
selected forward horizon. It does NOT stitch multiple snapshots into a
single continuous portfolio (that is a separate, much simpler mechanism --
the project owner's own "Połączone Portfolio" tab, plain replacement at
each snapshot's creation date, no smoothing). Stitching snapshots together
before running Sobol would confound "which parameter caused this" with
"which month had different market conditions" -- the two are not
separable once mixed, which is exactly why Sobol needs one fixed set of
fundamentals/prices per run.

For a FIXED snapshot (fixed fundamental_inputs, fixed price history up to
and including the snapshot's own creation date) and a FIXED forward
horizon (21/42/63/126 trading sessions -- 1/2/3/6 months), each Sobol
"run" is:
    1. Build mu_i for every ticker from THIS SNAPSHOT's fundamental_inputs,
       using the candidate parameter set's (alpha, gamma, kappa, n_ref).
    2. Run the SAME solver as everywhere else in this project
       (run_optimization_with_singleton_split) with the candidate's
       (lambda, nu, w_max, Rf) to get a weight vector.
    3. HOLD that weight vector fixed and evaluate it against REAL,
       ALREADY-REALIZED subsequent daily returns for exactly `horizon_days`
       sessions following the snapshot's creation date (walk-forward, no
       look-ahead -- the same discipline already enforced in the Relative
       Value pair overlay, ui/tab5_sandbox.py).
    4. Report three outputs from that one held horizon: CAGR, Sortino, and
       Max Drawdown.

Three outputs, not one, because a single metric hides exactly the failure
mode this whole analysis exists to catch: a parameter can look attractive
on CAGR while quietly increasing risk (Sortino falls or turns negative,
drawdown widens) -- comparing Sobol indices ACROSS the three outputs for
the same parameter is itself diagnostic (confirmed design intent, project
owner's own gamma-at-maximum observation that motivated this whole
feature).

Multiprocessing: `evaluate_single_parameter_set` is a module-level,
picklable function taking only plain data (dicts, DataFrames, floats) --
no closures, no Dash/UI state -- specifically so it can be dispatched to a
`concurrent.futures.ProcessPoolExecutor` sized to `os.cpu_count()`. Every
one of the N*(2k+2) (Sobol, calc_second_order=True) or r*(k+1) (Morris)
evaluations for one snapshot/horizon is fully independent of every other,
so this is an embarrassingly-parallel workload -- confirmed as the reason
this project's existing solver calls never visibly used more than one CPU
core: no single task before this one was ever large enough to justify it.
"""

from __future__ import annotations

import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from SALib.sample import sobol as sobol_sample_mod
from SALib.sample import morris as morris_sample_mod
from SALib.analyze import sobol as sobol_analyze_mod
from SALib.analyze import morris as morris_analyze_mod

from engine.returns import compute_composite_upside_row
from engine.risk import compute_estrada_matrix, compute_crash_overlap_matrix
from engine.optimizer import run_optimization_with_singleton_split
from engine.evaluation import evaluate_portfolio_performance

TRADING_DAYS_PER_YEAR = 252

# Confirmed default search ranges (2026-09-19) -- deliberately wide, matching
# the existing UI sliders' own min/max (STAGE4A_PARAMS_CONFIG, ui/components.py)
# rather than inventing new bounds. The project owner can override any of
# these per-run from the UI; these are only the pre-filled defaults.
DEFAULT_PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    "alpha":  (0.0, 1.0),
    "lambda": (0.1, 25.0),
    "nu":     (-5.0, 15.0),
    "gamma":  (0.0, 5.0),
    "kappa":  (0.0, 3.0),
    "w_max":  (0.05, 1.0),
    "rf":     (0.0, 0.25),
    "n_ref":  (1.0, 30.0),
}
PARAM_ORDER = ["alpha", "lambda", "nu", "gamma", "kappa", "w_max", "rf", "n_ref"]

HORIZON_DAYS = {"1m": 21, "2m": 42, "3m": 63, "6m": 126}


def _build_problem(param_ranges: Dict[str, Tuple[float, float]], n_tickers: Optional[int] = None) -> Dict[str, object]:
    """
    SALib 'problem' spec, in the fixed PARAM_ORDER (order matters -- SALib
    samples/analyzes columns positionally, not by name).

    CONFIRMED FIX (2026-09-19, found during initial testing): with `n_tickers`
    holdings, a portfolio needs w_max*n_tickers >= 1.0 to be feasible at
    all -- below that, EVERY candidate weight vector is mechanically
    infeasible regardless of any other parameter, which is not a genuine
    "risk" finding, just wasted, meaningless compute that pollutes the
    variance decomposition with content-free failures. When `n_tickers` is
    given, the w_max lower bound is clamped up to whichever is larger: the
    caller's own requested minimum, or 1.05/n_tickers (a small safety
    margin above the exact feasibility boundary, since SLSQP can still
    struggle to converge exactly AT the boundary even when technically
    feasible).
    """
    bounds = []
    for p in PARAM_ORDER:
        lo, hi = param_ranges.get(p, DEFAULT_PARAM_RANGES[p])
        if p == "w_max" and n_tickers and n_tickers > 0:
            lo = max(lo, 1.05 / n_tickers)
            hi = max(hi, lo + 1e-6)  # confirmed guard: never let a bad manual override invert the bounds
        bounds.append([lo, hi])
    return {"num_vars": len(PARAM_ORDER), "names": PARAM_ORDER, "bounds": bounds}


def evaluate_single_parameter_set(
    param_row: np.ndarray,
    fundamental_inputs: List[Dict[str, object]],
    cluster_of: Dict[str, int],
    risk_prices: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> Dict[str, float]:
    """
    ONE isolated, picklable evaluation -- the unit of work dispatched to the
    process pool. `param_row` is a length-8 array in PARAM_ORDER
    (alpha, lambda, nu, gamma, kappa, w_max, rf, n_ref).

    `risk_prices`: raw price history up to and including the snapshot's own
    creation date (same ~5y window convention as ui/tab5_sandbox.py's
    `risk_prices_full` -- used ONLY to estimate Sigma_eps/K, never touches
    the forward evaluation).

    `forward_returns`: REAL, already-realized daily log returns for the
    sessions strictly AFTER the snapshot's creation date, already sliced to
    exactly the horizon being tested by the caller (this function does not
    know or care which horizon it's evaluating -- it just uses whatever
    forward_returns it's handed). No look-ahead beyond what's already in
    this DataFrame is possible from inside this function.

    Sortino's MAR (minimum acceptable return) is fixed at 0, deliberately
    NOT tied to the candidate's own `rf` value -- confirmed design choice:
    since `rf` is itself one of the 8 swept parameters, using it as MAR
    would mechanically entangle the OUTPUT metric's definition with an
    INPUT being tested, biasing rf's own apparent sensitivity for reasons
    having nothing to do with portfolio construction.

    Returns dict: {"CAGR": float, "Sortino": float, "MaxDrawdown": float,
    "solver_success": bool} -- MaxDrawdown reported as a positive magnitude
    (e.g. 0.12 for -12%). On any failure (infeasible solver, degenerate
    horizon), returns NaN for the three metrics and solver_success=False
    rather than raising -- a single bad sample must not crash the whole
    batch.
    """
    alpha, lam, nu, gamma, kappa, w_max, rf, n_ref = [float(x) for x in param_row]

    try:
        tickers = list(risk_prices.columns)
        fund_by_ticker = {r["Ticker"]: r for r in fundamental_inputs}
        mu_vec = pd.Series({
            t: compute_composite_upside_row(fund_by_ticker.get(t, {}), gamma, kappa, n_ref, alpha=alpha)["mu_i"]
            for t in tickers
        })

        returns_window = np.log(risk_prices / risk_prices.shift(1)).dropna()
        usable_tickers = list(returns_window.columns)
        clusters_dict: Dict[int, List[str]] = {}
        for t in usable_tickers:
            clusters_dict.setdefault(cluster_of.get(t, 1), []).append(t)

        sigma_eps = compute_estrada_matrix(returns_window)
        crash = compute_crash_overlap_matrix(risk_prices[usable_tickers], quantile=0.10)
        split_result = run_optimization_with_singleton_split(
            mu_vec.reindex(usable_tickers), sigma_eps, crash["K"], clusters_dict,
            lam, w_max=w_max, Rf=rf, nu=nu,
        )
        weights = split_result["final"]["weights"].reindex(tickers).fillna(0.0)
        if weights.sum() <= 1e-9:
            raise ValueError("degenerate zero weight vector")
        weights = weights / weights.sum()

        perf = evaluate_portfolio_performance(weights, forward_returns, is_log_returns=True)
        daily = np.array(perf["portfolio_returns"], dtype=float)
        if daily.size == 0:
            raise ValueError("empty forward return window")

        n_days = daily.size
        cum_return = float(np.prod(1.0 + daily) - 1.0)
        cagr = float((1.0 + cum_return) ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0)

        downside = daily[daily < 0.0]
        downside_dev = float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(TRADING_DAYS_PER_YEAR)) if downside.size > 0 else 0.0
        mean_ann = float(daily.mean() * TRADING_DAYS_PER_YEAR)
        sortino = (mean_ann - 0.0) / downside_dev if downside_dev > 1e-12 else float("nan")

        max_dd = float(-min(perf["drawdown_series"])) if perf["drawdown_series"] else 0.0

        return {"CAGR": cagr, "Sortino": sortino, "MaxDrawdown": max_dd, "solver_success": True}
    except Exception:
        return {"CAGR": float("nan"), "Sortino": float("nan"), "MaxDrawdown": float("nan"), "solver_success": False}


def _run_batch_parallel(
    sample: np.ndarray,
    fundamental_inputs: List[Dict[str, object]],
    cluster_of: Dict[str, int],
    risk_prices: pd.DataFrame,
    forward_returns: pd.DataFrame,
    n_workers: Optional[int] = None,
    on_progress: Optional[Callable[[int, int, float], None]] = None,
) -> pd.DataFrame:
    """
    Dispatches every row of `sample` to a ProcessPoolExecutor sized to
    `n_workers` (defaults to os.cpu_count() -- confirmed 2026-09-19 for a
    32-thread i9-14900KF, but reads the actual machine at call time rather
    than hardcoding that number, so this stays correct on any hardware).

    `risk_prices` and `forward_returns` are frozen, read-only for the whole
    batch (same snapshot, same horizon, every candidate parameter set) --
    confirmed reason NOT to re-fetch/re-slice per-row: with N in the
    thousands, doing so would repeat identical work thousands of times for
    zero benefit, since none of these three inputs vary across rows.

    `on_progress(completed, total, elapsed_seconds)` is called after every
    completed future, throttled by the caller (not here) if needed -- for
    tens of thousands of rows, updating a live UI status string after
    EVERY single completion would make the progress reporting itself a
    non-trivial fraction of the runtime.
    """
    n_workers = n_workers or os.cpu_count() or 1
    total = len(sample)
    results: List[Optional[Dict[str, float]]] = [None] * total
    start = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(evaluate_single_parameter_set, row, fundamental_inputs, cluster_of, risk_prices, forward_returns): i
            for i, row in enumerate(sample)
        }
        completed = 0
        for future in as_completed(futures):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception:
                results[i] = {"CAGR": float("nan"), "Sortino": float("nan"), "MaxDrawdown": float("nan"), "solver_success": False}
            completed += 1
            if on_progress is not None:
                on_progress(completed, total, time.time() - start)

    return pd.DataFrame(results)


def run_morris_screen(
    fundamental_inputs: List[Dict[str, object]],
    cluster_of: Dict[str, int],
    risk_prices: pd.DataFrame,
    forward_returns: pd.DataFrame,
    param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    n_trajectories: int = 20,
    n_workers: Optional[int] = None,
    on_progress: Optional[Callable[[int, int, float], None]] = None,
) -> Dict[str, object]:
    """
    Optional, cheap screen BEFORE a full Sobol run -- confirmed opt-in, not
    forced (2026-09-19): r*(k+1) evaluations (r=20, k=8 -> 180 runs) versus
    Sobol's N*(2k+2) (>=4600 at N=256). Reports mu_star (mean absolute
    elementary effect -- overall importance) and sigma (std of elementary
    effects -- high sigma relative to mu_star suggests nonlinearity or
    interaction with other parameters) for each of the three outputs.
    """
    param_ranges = param_ranges or DEFAULT_PARAM_RANGES
    problem = _build_problem(param_ranges, n_tickers=risk_prices.shape[1])
    sample = morris_sample_mod.sample(problem, N=n_trajectories, num_levels=4)

    raw = _run_batch_parallel(sample, fundamental_inputs, cluster_of, risk_prices, forward_returns, n_workers, on_progress)

    results = {}
    for output_name in ["CAGR", "Sortino", "MaxDrawdown"]:
        y = raw[output_name].values
        y_filled = np.nan_to_num(y, nan=np.nanmedian(y) if np.isfinite(y).any() else 0.0)
        try:
            morris_result = morris_analyze_mod.analyze(problem, sample, y_filled, num_levels=4)
            results[output_name] = {
                "names": list(morris_result["names"]),
                "mu_star": [float(v) for v in morris_result["mu_star"]],
                "sigma": [float(v) for v in morris_result["sigma"]],
            }
        except Exception as e:
            results[output_name] = {"error": str(e)}

    return {
        "outputs": results, "n_runs": len(sample),
        "n_solver_failures": int((~raw["solver_success"]).sum()),
        "param_order": PARAM_ORDER,
    }


def run_sobol_batch(
    fundamental_inputs: List[Dict[str, object]],
    cluster_of: Dict[str, int],
    risk_prices: pd.DataFrame,
    forward_returns: pd.DataFrame,
    param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    N: int = 256,
    n_workers: Optional[int] = None,
    on_progress: Optional[Callable[[int, int, float], None]] = None,
) -> Dict[str, object]:
    """
    Full Sobol sensitivity analysis for ONE frozen snapshot / ONE forward
    horizon. N*(2*8+2) = N*18 total solver+forward-track evaluations
    (calc_second_order=True, confirmed 2026-09-19 -- keeps second-order
    interaction indices available, e.g. gamma-nu interaction, directly
    relevant to the project owner's own instability question from earlier
    in this project).

    Also computes PRCC (Partial Rank Correlation Coefficient) on the SAME
    sample at effectively no extra cost (zero additional solver calls) --
    confirmed as a cheap, independent cross-check: if PRCC's parameter
    ranking agrees with Sobol's S_i ranking, that strengthens confidence in
    both; disagreement flags a likely non-monotonic relationship worth a
    second look before trusting either in isolation.

    Returns dict:
        "sobol"      : {output_name: {"S1": [...], "S1_conf": [...], "ST": [...], "ST_conf": [...],
                                       "S2": [[...]], "S2_conf": [[...]]}}
        "prcc"       : {output_name: {param_name: float}}
        "param_order": PARAM_ORDER
        "n_runs", "n_solver_failures"
        "raw_sample" : the full (sample, outputs) pair, kept SERVER-SIDE only
                       (the caller is responsible for NOT shipping this to
                       the browser via dcc.Store -- confirmed design,
                       thousands of rows is too much for a JSON round-trip)
    """
    param_ranges = param_ranges or DEFAULT_PARAM_RANGES
    problem = _build_problem(param_ranges, n_tickers=risk_prices.shape[1])
    sample = sobol_sample_mod.sample(problem, N=N, calc_second_order=True)

    raw = _run_batch_parallel(sample, fundamental_inputs, cluster_of, risk_prices, forward_returns, n_workers, on_progress)

    sobol_results = {}
    prcc_results = {}
    for output_name in ["CAGR", "Sortino", "MaxDrawdown"]:
        y = raw[output_name].values
        n_valid = int(np.isfinite(y).sum())
        if n_valid < len(y):
            # Confirmed fallback (not a silent corruption): a failed row keeps its NaN out of
            # the variance-decomposition math by substituting the sample median -- flagged
            # via n_solver_failures below rather than hidden, so the UI can warn if this
            # fraction is large enough to cast doubt on the result.
            fill_value = np.nanmedian(y) if n_valid > 0 else 0.0
            y = np.nan_to_num(y, nan=fill_value)
        try:
            sobol_out = sobol_analyze_mod.analyze(problem, y, calc_second_order=True, print_to_console=False)
            sobol_results[output_name] = {
                "names": PARAM_ORDER,
                "S1": [float(v) for v in sobol_out["S1"]], "S1_conf": [float(v) for v in sobol_out["S1_conf"]],
                "ST": [float(v) for v in sobol_out["ST"]], "ST_conf": [float(v) for v in sobol_out["ST_conf"]],
                "S2": [[(float(v) if np.isfinite(v) else None) for v in row] for row in sobol_out["S2"]],
            }
        except Exception as e:
            sobol_results[output_name] = {"error": str(e)}

        prcc_results[output_name] = _compute_prcc(sample, y)

    return {
        "sobol": sobol_results, "prcc": prcc_results, "param_order": PARAM_ORDER,
        "n_runs": len(sample), "n_solver_failures": int((~raw["solver_success"]).sum()),
        "raw_sample": sample, "raw_outputs": raw,
    }


def _compute_prcc(sample: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Partial Rank Correlation Coefficient for each parameter against output
    `y`, on the SAME sample already drawn for Sobol -- zero extra solver
    calls (confirmed design: reuses the existing (sample, y) pair passed
    in by the caller).

    Method: rank-transform every column (parameters + y), then for
    parameter i, regress its ranks and y's ranks each against the ranks of
    ALL OTHER parameters, and correlate the two residuals -- the standard
    partial-correlation-via-residuals construction, applied to ranks
    instead of raw values so the result is robust to monotonic
    nonlinearity (not just linear effects).
    """
    from scipy.stats import rankdata, pearsonr

    ranked_x = np.column_stack([rankdata(sample[:, j]) for j in range(sample.shape[1])])
    ranked_y = rankdata(y)

    prcc = {}
    for i, name in enumerate(PARAM_ORDER):
        other_idx = [j for j in range(len(PARAM_ORDER)) if j != i]
        X_other = ranked_x[:, other_idx]
        X_other_design = np.column_stack([np.ones(len(X_other)), X_other])

        beta_xi, *_ = np.linalg.lstsq(X_other_design, ranked_x[:, i], rcond=None)
        resid_xi = ranked_x[:, i] - X_other_design @ beta_xi

        beta_y, *_ = np.linalg.lstsq(X_other_design, ranked_y, rcond=None)
        resid_y = ranked_y - X_other_design @ beta_y

        if np.std(resid_xi) < 1e-9 or np.std(resid_y) < 1e-9:
            prcc[name] = float("nan")
        else:
            r, _ = pearsonr(resid_xi, resid_y)
            prcc[name] = float(r)
    return prcc


def estimate_run_cost(N: int, k: int = 8, seconds_per_eval: float = 0.05, n_workers: Optional[int] = None) -> Dict[str, float]:
    """UI helper: 'N=256 -> 4608 runs, ~X seconds on Y workers' -- shown BEFORE
    the user commits to a run, confirmed requirement (2026-09-19)."""
    n_workers = n_workers or os.cpu_count() or 1
    n_runs = N * (2 * k + 2)
    return {
        "n_runs": n_runs,
        "est_seconds_sequential": n_runs * seconds_per_eval,
        "est_seconds_parallel": (n_runs * seconds_per_eval) / max(n_workers, 1),
        "n_workers": n_workers,
    }
