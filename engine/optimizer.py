"""
engine/optimizer.py
====================
True Two-Stage SLSQP portfolio optimizer (Section 3.4) + the iterative
Dynamic Singleton Split / Binding Ceiling Throttle procedure (Section 3.4
extension), plus the standalone diagnostic Asset-Level Tail-Penalized
Sortino ratio.

No Dash / UI / disk-I/O code lives here on purpose. Moved out of the old
monolithic tps_solver.py (Etap 0 architecture split, PROJECT_CONTEXT.md)
with NO behavior change -- every function below is a verbatim relocation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

EPSILON = 1e-6  # safety epsilon added to TPS denominators to avoid division by zero


def compute_asset_tps(
    mu_vec: pd.Series,
    P_vec: pd.Series,
    semideviation_ann_vec: pd.Series,
    Rf: float = 0.045,
) -> pd.Series:
    """
    Per-asset Tail-Penalized Sortino ratio (Section 4.4):

        TPS_i = (mu_i - Rf) / (semideviation_ann_i * P_i + epsilon)

    Parameters
    ----------
    mu_vec : pd.Series
        Composite Forward Upside (mu_i) per ticker (Section 4.2 output).
    P_vec : pd.Series
        Exponential tail penalty multiplier (P_i) per ticker (Section 4.1/4.3 output).
    semideviation_ann_vec : pd.Series
        Annualized downside semi-deviation per ticker -- typically
        `semideviation_ann_from_matrix(downside_cov_matrix)`.
    Rf : float, default 0.045
        Annualized risk-free rate (hurdle rate) subtracted from mu_i.

    Returns
    -------
    pd.Series
        TPS_i per ticker, aligned on the intersection of all three input indices.
    """
    idx = mu_vec.index.intersection(P_vec.index).intersection(semideviation_ann_vec.index)
    if len(idx) == 0:
        raise ValueError("compute_asset_tps: no common tickers across mu_vec/P_vec/semideviation_ann_vec.")

    mu = mu_vec.reindex(idx)
    P = P_vec.reindex(idx)
    sd = semideviation_ann_vec.reindex(idx)

    denom = sd * P + EPSILON
    tps = (mu - Rf) / denom
    return tps.reindex(idx)


# ---------------------------------------------------------------------------
# 3. TRUE TWO-STAGE SLSQP OPTIMIZATION ENGINE (Section 3.4)
# ---------------------------------------------------------------------------
# Replaces the old heuristic power-weighted allocation (`_power_weighted_normalize`
# / `run_two_stage_allocation`) AND the old single-pass linear-penalty SLSQP
# refinement (`optimize_portfolio_slsqp` / `_portfolio_tps_neg`), both removed.
#
# TPS is now treated as an optimization OBJECTIVE over weight vectors, not a
# per-asset scalar used to hand-weight a heuristic -- per Section 3.4:
#
#   TPS(w) = (w.mu - Rf) / (sqrt(w.Sigma_eps.w) * exp(lambda * sqrt(w.K.w)) + eps)
#
# Stage 1 solves this INDEPENDENTLY per cluster over intra-cluster weights g
# (dimension = cluster size, sums to 1 within the cluster). Stage 2 then solves
# it GLOBALLY over inter-cluster weights W (dimension = number of clusters,
# sums to 1 across the whole portfolio), reconstructing the full N-asset weight
# vector as w_final,i = W_{cluster(i)} * g*_i with g* FROZEN from Stage 1 --
# but Stage 2's quadratic risk/crash terms still use the FULL N x N Sigma_eps
# and K matrices (cross-cluster covariance and cross-cluster crash coupling
# matter and are not discarded just because the decision variable is smaller).
#
# The old asset-level P_i (Z-score tail penalty) plays NO role here -- this
# engine only consumes Sigma_eps and K. P_i remains a separate, purely
# diagnostic/visual quantity computed elsewhere (see `compute_tail_penalty` in
# quant_terminal.py and the Tab 3 underwater chart) and is never passed in.
#
# A nice side effect of using a real solver: the old heuristic's "equal-weight
# fallback if every TPS_i <= 0" special case (needed because `x^power` is
# degenerate for non-positive x) no longer exists -- SLSQP's objective is a
# smooth function of w regardless of the sign of mu_i, so there is nothing
# special to special-case.

def _quad_form_sqrt(w: np.ndarray, M: np.ndarray) -> float:
    """sqrt(w.T @ M @ w), clipped at 0 to guard against tiny negative values
    from floating-point noise on an otherwise-PSD matrix."""
    return float(np.sqrt(max(float(w @ M @ w), 0.0)))


def _tps_neg(w: np.ndarray, mu_arr: np.ndarray, sigma_arr: np.ndarray, k_arr: np.ndarray, lam: float, Rf: float) -> float:
    """Shared objective for BOTH stages: -TPS(w) (scipy minimizes -> maximizes TPS)."""
    mu_p = float(w @ mu_arr)
    sigma_p = _quad_form_sqrt(w, sigma_arr)
    k_p = _quad_form_sqrt(w, k_arr)
    tps = (mu_p - Rf) / (sigma_p * np.exp(lam * k_p) + EPSILON)
    return -tps


def run_stage1_intra_cluster_slsqp(
    mu_vec: pd.Series,
    sigma_eps: pd.DataFrame,
    K_matrix: pd.DataFrame,
    clusters_dict: Dict[object, List[str]],
    lam: float,
    Rf: float = 0.045,
    maxiter: int = 300,
) -> Dict[str, object]:
    """
    Section 3.4, Stage 1 (Intra-Cluster SLSQP).

    For each cluster C_k independently:
        max_g   TPS_Ck(g) = (g.mu_Ck - Rf) / (sqrt(g.Sigma_eps,Ck.g) * exp(lam*sqrt(g.K_Ck.g)) + eps)
        s.t.    sum(g) = 1,  0 <= g_i <= 1

    Sub-matrices `Sigma_eps[C_k, C_k]` and `K[C_k, C_k]` are sliced from the
    full-universe matrices -- within-cluster crash coupling and covariance are
    fully captured; cross-cluster terms are intentionally excluded here (they
    only enter in Stage 2, where the full matrices are used against the
    reconstructed portfolio-wide weight vector).

    Single-asset clusters are handled as a trivial g=[1.0] case (skips SLSQP
    entirely -- nothing to optimize with one degree of freedom under a
    sum-to-1 constraint).

    Parameters
    ----------
    mu_vec : pd.Series
        Composite Forward Upside per ticker, must cover every ticker in
        `clusters_dict`.
    sigma_eps : pd.DataFrame
        Full N x N regularized Estrada downside semi-covariance matrix
        (`compute_estrada_matrix` output, ridge already applied).
    K_matrix : pd.DataFrame
        Full N x N Discrete Crash-Overlap Matrix (`compute_crash_overlap_matrix`
        output's "K").
    clusters_dict : dict
        Mapping {cluster_id: [ticker, ...]}.
    lam : float
        Lambda -- crash-overlap penalty sensitivity (Section 3.3.D / 3.4).
    Rf : float, default 0.045
        Annualized risk-free hurdle rate.
    maxiter : int, default 300
        Passed through to `scipy.optimize.minimize`.

    Returns
    -------
    dict with keys:
        "g"                : pd.Series, full-universe intra-cluster weights
                              (sums to 1 WITHIN each cluster, not across the
                              whole portfolio -- Stage 2 handles that).
        "cluster_mu"       : dict {cluster_id: g*.mu_Ck}
        "cluster_sigma"    : dict {cluster_id: sqrt(g*.Sigma_Ck.g*)}
        "cluster_k_penalty": dict {cluster_id: sqrt(g*.K_Ck.g*)}
        "cluster_tps"      : dict {cluster_id: TPS_Ck at the optimum}
        "cluster_success"  : dict {cluster_id: bool}
        "cluster_message"  : dict {cluster_id: str}
    """
    g_parts: List[pd.Series] = []
    cluster_mu, cluster_sigma, cluster_kpen, cluster_tps = {}, {}, {}, {}
    cluster_success, cluster_message = {}, {}

    for ck, members in clusters_dict.items():
        members = list(members)
        n = len(members)
        mu_arr = mu_vec.reindex(members).values.astype(float)
        sigma_arr = sigma_eps.loc[members, members].values.astype(float)
        k_arr = K_matrix.loc[members, members].values.astype(float)

        if n == 1:
            g_opt = np.array([1.0])
            success, message = True, "Single-asset cluster (trivial, SLSQP skipped)."
        else:
            x0 = np.full(n, 1.0 / n)
            bounds = [(0.0, 1.0) for _ in range(n)]
            constraints = [{"type": "eq", "fun": lambda g: np.sum(g) - 1.0}]
            result = minimize(
                _tps_neg, x0, args=(mu_arr, sigma_arr, k_arr, lam, Rf),
                method="SLSQP", bounds=bounds, constraints=constraints,
                options={"maxiter": maxiter, "ftol": 1e-10},
            )
            g_opt = np.clip(result.x, 0.0, 1.0)
            total = g_opt.sum()
            g_opt = g_opt / total if total > 1e-12 else np.full(n, 1.0 / n)
            success, message = bool(result.success), str(result.message)

        g_parts.append(pd.Series(g_opt, index=members))
        cluster_mu[ck] = float(g_opt @ mu_arr)
        cluster_sigma[ck] = _quad_form_sqrt(g_opt, sigma_arr)
        cluster_kpen[ck] = _quad_form_sqrt(g_opt, k_arr)
        cluster_tps[ck] = (cluster_mu[ck] - Rf) / (cluster_sigma[ck] * np.exp(lam * cluster_kpen[ck]) + EPSILON)
        cluster_success[ck] = success
        cluster_message[ck] = message

    return {
        "g": pd.concat(g_parts),
        "cluster_mu": cluster_mu,
        "cluster_sigma": cluster_sigma,
        "cluster_k_penalty": cluster_kpen,
        "cluster_tps": cluster_tps,
        "cluster_success": cluster_success,
        "cluster_message": cluster_message,
    }


def run_stage2_inter_cluster_slsqp(
    mu_vec: pd.Series,
    sigma_eps: pd.DataFrame,
    K_matrix: pd.DataFrame,
    clusters_dict: Dict[object, List[str]],
    g_star: pd.Series,
    lam: float,
    w_max: float = 0.20,
    Rf: float = 0.045,
    maxiter: int = 300,
) -> Dict[str, object]:
    """
    Section 3.4, Stage 2 (Inter-Cluster SLSQP).

        max_W   TPS_P(W) = (w_final.mu - Rf) / (sqrt(w_final.Sigma_eps.w_final) * exp(lam*sqrt(w_final.K.w_final)) + eps)
        s.t.    sum(W_Ck) = 1,  W_Ck >= 0,  w_final,i = W_Ck * g*_{k,i} <= w_max  (for all i)

    `g_star` (Stage 1 output) is FROZEN here -- only the M-dimensional cluster
    weight vector W is optimized. `w_final` is reconstructed each objective
    evaluation as `g_star * W[cluster_of(i)]` and plugged into the FULL N x N
    `sigma_eps`/`K_matrix` (cross-cluster terms matter and are not reduced away).

    Box constraint as bounds, not an inequality constraint (same design choice
    as the old `optimize_portfolio_slsqp`, kept for the same reason -- bounds
    are more numerically stable for SLSQP than a matching inequality
    constraint): the per-asset cap `W_Ck * g*_i <= w_max` is tightest for
    whichever member of C_k has the largest g*_i, so each cluster's upper
    bound on W_Ck is `min(1.0, w_max / max_i(g*_{k,i}))`.

    Parameters
    ----------
    mu_vec, sigma_eps, K_matrix : see `run_stage1_intra_cluster_slsqp`.
    clusters_dict : dict
        Mapping {cluster_id: [ticker, ...]} -- SAME clustering used in Stage 1.
    g_star : pd.Series
        Stage 1's frozen intra-cluster weights (full universe).
    lam : float
        Same lambda as Stage 1 (Section 3.3.D / 3.4) -- kept consistent across
        both stages so the crash-penalty sensitivity means the same thing
        throughout the pipeline.
    w_max : float, default 0.20
        Hard per-asset concentration cap.
    Rf : float, default 0.045
        Annualized risk-free hurdle rate.
    maxiter : int, default 300
        Passed through to `scipy.optimize.minimize`.

    Returns
    -------
    dict with keys:
        "weights"      : pd.Series, final w_final (sums to 1.0, each in [0, w_max])
        "cluster_W"    : dict {cluster_id: W_Ck at the optimum}
        "success"      : bool, solver convergence flag
        "message"      : str, solver status message
        "mu_p"         : float, w_final . mu at the optimum
        "delta_p"      : float, sqrt(w_final . Sigma_eps . w_final) at the optimum
        "k_penalty_p"  : float, sqrt(w_final . K . w_final) at the optimum
        "tps_p"        : float, final portfolio TPS at the optimum

    Raises
    ------
    ValueError
        If the box constraint is infeasible given Stage 1's intra-cluster
        concentration, i.e. the per-cluster upper bounds on W_Ck can't sum to
        1.0 (mirrors the old `optimize_portfolio_slsqp`'s `N * w_max < 1.0`
        check, generalized to the cluster level).
    """
    cluster_ids = list(clusters_dict.keys())
    M = len(cluster_ids)
    tickers = [t for ck in cluster_ids for t in clusters_dict[ck]]
    N = len(tickers)

    cluster_idx_arr = np.array(
        [ci for ci, ck in enumerate(cluster_ids) for _ in clusters_dict[ck]], dtype=int
    )

    g_full_arr = g_star.reindex(tickers).values.astype(float)
    mu_arr = mu_vec.reindex(tickers).values.astype(float)
    sigma_arr = sigma_eps.loc[tickers, tickers].values.astype(float)
    k_arr = K_matrix.loc[tickers, tickers].values.astype(float)

    # Per-cluster upper bound on W_Ck implied by the per-asset w_max box constraint.
    upper_bounds = []
    for ck in cluster_ids:
        max_g = float(g_star.reindex(clusters_dict[ck]).max())
        ub = min(1.0, w_max / max_g) if max_g > 1e-12 else 1.0
        upper_bounds.append(ub)

    if sum(upper_bounds) < 1.0 - 1e-9:
        raise ValueError(
            f"run_stage2_inter_cluster_slsqp: infeasible constraints -- w_max={w_max} is too tight "
            f"given Stage 1's intra-cluster concentration (sum of implied per-cluster caps = "
            f"{sum(upper_bounds):.4f} < 1.0). Raise w_max."
        )

    x0 = np.minimum(np.full(M, 1.0 / M), upper_bounds)
    if x0.sum() > 1e-12:
        x0 = x0 / x0.sum()
        x0 = np.minimum(x0, upper_bounds)
        if x0.sum() > 1e-12:
            x0 = x0 / x0.sum()
    else:
        x0 = np.array(upper_bounds) / sum(upper_bounds)

    bounds = [(0.0, ub) for ub in upper_bounds]
    constraints = [{"type": "eq", "fun": lambda W: np.sum(W) - 1.0}]

    def objective(W):
        w_final = g_full_arr * W[cluster_idx_arr]
        return _tps_neg(w_final, mu_arr, sigma_arr, k_arr, lam, Rf)

    result = minimize(
        objective, x0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": maxiter, "ftol": 1e-10},
    )

    W_opt = np.clip(result.x, 0.0, upper_bounds)
    total = W_opt.sum()
    W_opt = W_opt / total if total > 1e-12 else np.array(upper_bounds) / sum(upper_bounds)

    w_final_arr = g_full_arr * W_opt[cluster_idx_arr]
    w_final_arr = np.clip(w_final_arr, 0.0, w_max)  # numerical-drift insurance
    total2 = w_final_arr.sum()
    w_final_arr = w_final_arr / total2 if total2 > 1e-12 else np.full(N, 1.0 / N)

    weights = pd.Series(w_final_arr, index=tickers)
    mu_p = float(w_final_arr @ mu_arr)
    delta_p = _quad_form_sqrt(w_final_arr, sigma_arr)
    k_p = _quad_form_sqrt(w_final_arr, k_arr)
    tps_p = (mu_p - Rf) / (delta_p * np.exp(lam * k_p) + EPSILON)

    return {
        "weights": weights,
        "cluster_W": dict(zip(cluster_ids, W_opt.tolist())),
        "success": bool(result.success),
        "message": str(result.message),
        "mu_p": mu_p,
        "delta_p": delta_p,
        "k_penalty_p": k_p,
        "tps_p": tps_p,
    }


def run_true_two_stage_optimization(
    mu_vec: pd.Series,
    sigma_eps: pd.DataFrame,
    K_matrix: pd.DataFrame,
    clusters_dict: Dict[object, List[str]],
    lam: float,
    w_max: float = 0.20,
    Rf: float = 0.045,
    maxiter: int = 300,
) -> Dict[str, object]:
    """
    Single entry point: Stage 1 -> Stage 2 (Section 3.4). Shared by the main
    Stage 4B solver AND the Tab 5 sandbox (`compute_sandbox_allocation`) in
    quant_terminal.py, so both consume identical optimization logic -- no
    separate "sandbox heuristic" path anymore.

    Returns dict merging both stages' outputs:
        "weights"        : pd.Series, final w_final
        "g"               : pd.Series, Stage 1's frozen intra-cluster weights
        "cluster_W"       : dict {cluster_id: W_Ck}
        "cluster_mu", "cluster_sigma", "cluster_k_penalty", "cluster_tps" : Stage 1 per-cluster diagnostics
        "stage1_success", "stage1_message" : dict {cluster_id: ...}
        "success", "message" : Stage 2 solver convergence flag/message
        "mu_p", "delta_p", "k_penalty_p", "tps_p" : Stage 2 portfolio-level metrics at the optimum
    """
    stage1 = run_stage1_intra_cluster_slsqp(mu_vec, sigma_eps, K_matrix, clusters_dict, lam, Rf=Rf, maxiter=maxiter)
    stage2 = run_stage2_inter_cluster_slsqp(mu_vec, sigma_eps, K_matrix, clusters_dict, stage1["g"], lam, w_max=w_max, Rf=Rf, maxiter=maxiter)

    return {
        "weights": stage2["weights"],
        "g": stage1["g"],
        "cluster_W": stage2["cluster_W"],
        "cluster_mu": stage1["cluster_mu"],
        "cluster_sigma": stage1["cluster_sigma"],
        "cluster_k_penalty": stage1["cluster_k_penalty"],
        "cluster_tps": stage1["cluster_tps"],
        "stage1_success": stage1["cluster_success"],
        "stage1_message": stage1["cluster_message"],
        "success": stage2["success"],
        "message": stage2["message"],
        "mu_p": stage2["mu_p"],
        "delta_p": stage2["delta_p"],
        "k_penalty_p": stage2["k_penalty_p"],
        "tps_p": stage2["tps_p"],
    }


# ---------------------------------------------------------------------------
# 4. DYNAMIC SINGLETON SPLIT (Iterative Multi-Pass)
# ---------------------------------------------------------------------------
# Structural fix for a genuine mathematical property of the "true" two-stage
# split: Stage 1 has NO knowledge of w_max (only 0 <= g_i <= 1, per spec), so
# it can legitimately concentrate a cluster's intra-cluster weight g_i on one
# dominant asset far past what Stage 2's single remaining degree of freedom
# per cluster (W_Ck) can then repair. This is common, not an edge case, in a
# hyper-growth universe where one name in a cluster often has much higher mu_i
# than its peers.
#
# IMPORTANT REVISION: an earlier version of this only triggered a split when
# Stage 2 was outright INFEASIBLE (couldn't reach sum(w)=1 at all). That missed
# a real, silent failure mode: a dominant ticker's g_i > w_max caps its WHOLE
# cluster's W_Ck below 1.0 (W_Ck <= w_max/g_i) even when the REST of the
# portfolio has plenty of slack elsewhere -- so the overall optimization stays
# formally "feasible" and converges without ever raising, while quietly
# throttling that dominant ticker's cluster-mates (e.g. MU at g_i=62.7% capping
# its whole cluster at W_C1<=55.8%, silently squeezing NVDA/AVGO) with no
# visible signal that anything unusual happened. The detection criterion below
# now fires on this "silent throttling" case too, not just hard infeasibility.
#
# Two-pass was also too rigid for CASCADING dominance: removing one cluster's
# dominant member can concentrate the remaining sub-cluster enough that a NEW
# member now exceeds w_max in turn. This is now handled by an actual `while`
# loop (bounded by `max_passes`, default = total asset count, a safe upper
# bound since at most every single asset could end up promoted one at a time).

def _identify_throttling_tickers(
    g_star: pd.Series, clusters_dict: Dict[object, List[str]], w_max: float,
    cluster_W: Optional[Dict[object, float]] = None, infeasible: bool = False, tol: float = 1e-3,
) -> List[str]:
    """
    "Binding Ceiling Throttle" criterion -- promotes a cluster's single most
    dominant member ONLY if BOTH hold:

      1. Concentration: g_i > w_max for that ticker -- the structural
         precondition. Without this, the cluster's own Stage-2 bound
         (w_max/max_g) is never below 1.0 in the first place, so there is
         nothing to throttle.
      2. Active binding: the cluster ACTUALLY hit its implied ceiling in
         Stage 2 -- `W_Ck >= w_max/max_g - tol` -- meaning the solver wanted
         to allocate more to this cluster but the dominant member's own cap
         stopped it. If the whole pass was infeasible, there's no solved
         `cluster_W` to check against, so infeasibility itself counts as
         binding by definition.

    Revision history / why this replaced a looser "g_i > w_max alone" check:
    that version fired on concentration ALONE, regardless of whether the
    cluster ever got enough portfolio-level demand to bump into its bound. A
    2-member cluster with a 60/40 intra-cluster split that only ends up
    getting, say, 20% of the total portfolio (dominant member = 12% overall,
    nowhere near w_max) would still get flagged and split -- destroying the
    cluster structure for no actual gain, since nothing was really being
    constrained. Requiring the SECOND condition (the cluster's W_Ck is
    genuinely pinned at its own bound) restricts promotion to cases where
    splitting actually unlocks capital that Stage 2 wanted to allocate but
    couldn't -- true "silent throttling" (Section 3.4 extension), not mere
    concentration.

    Only the cluster's single most-dominant member is ever flagged per
    cluster per pass (promoting it is what relaxes that cluster's bound;
    should a NEW dominant member emerge among what's left, the next pass's
    own check catches it -- this is what makes the outer loop iterative
    rather than a fixed two passes).

    Restricted to clusters with MORE than one member throughout -- an
    existing singleton always has g_i=1.0 trivially, but there's no cluster
    left to throttle and nothing further to split off.
    """
    dominant = []
    for ck, members in clusters_dict.items():
        if len(members) <= 1:
            continue

        top_ticker = max(members, key=lambda t: float(g_star.get(t, 0.0)))
        max_g = float(g_star.get(top_ticker, 0.0))
        if max_g <= w_max:
            continue  # criterion 1 fails -- no structural dominance in this cluster at all

        if infeasible:
            dominant.append(top_ticker)  # no W_Ck to check -- infeasibility itself is binding
            continue

        if cluster_W is None:
            continue  # defensive: shouldn't happen on a feasible pass, but never guess without data

        implied_bound = min(1.0, w_max / max_g) if max_g > 1e-12 else 1.0
        w_ck = float(cluster_W.get(ck, 0.0))
        if w_ck >= implied_bound - tol:
            dominant.append(top_ticker)  # criterion 2 confirmed -- cluster is genuinely pinned at its ceiling

    return dominant


def _build_split_clusters(clusters_dict: Dict[object, List[str]], dominant_tickers: List[str]) -> Dict[object, List[str]]:
    """
    Carves `dominant_tickers` out of their current clusters into their own
    singleton clusters (key `f"SINGLETON_{ticker}"`), leaving the remaining
    members of each cluster grouped as before. A cluster left with zero
    remaining members simply disappears from the new dict. Safe to call
    repeatedly across passes: pre-existing singleton clusters whose member
    isn't in `dominant_tickers` this round pass through unchanged.
    """
    dominant_set = set(dominant_tickers)
    new_clusters: Dict[object, List[str]] = {}
    for ck, members in clusters_dict.items():
        remaining = [t for t in members if t not in dominant_set]
        if remaining:
            new_clusters[ck] = remaining
    for t in dominant_tickers:
        new_clusters[f"SINGLETON_{t}"] = [t]
    return new_clusters


def _run_single_pass(mu_vec, sigma_eps, K_matrix, clusters_dict, lam, w_max, Rf, maxiter):
    """
    One Stage1+Stage2 attempt, normalizing both outcomes (feasible / infeasible)
    into the same dict shape so the history log and the detection step don't
    need to special-case which one happened.
    """
    try:
        result = run_true_two_stage_optimization(mu_vec, sigma_eps, K_matrix, clusters_dict, lam, w_max=w_max, Rf=Rf, maxiter=maxiter)
        return result, False
    except ValueError as e:
        stage1_only = run_stage1_intra_cluster_slsqp(mu_vec, sigma_eps, K_matrix, clusters_dict, lam, Rf=Rf, maxiter=maxiter)
        result = {
            "weights": None, "success": False, "message": str(e),
            "g": stage1_only["g"], "cluster_mu": stage1_only["cluster_mu"],
            "cluster_sigma": stage1_only["cluster_sigma"], "cluster_k_penalty": stage1_only["cluster_k_penalty"],
            "cluster_tps": stage1_only["cluster_tps"], "stage1_success": stage1_only["cluster_success"],
            "stage1_message": stage1_only["cluster_message"],
            "mu_p": None, "delta_p": None, "k_penalty_p": None, "tps_p": None,
        }
        return result, True


def run_optimization_with_singleton_split(
    mu_vec: pd.Series,
    sigma_eps: pd.DataFrame,
    K_matrix: pd.DataFrame,
    clusters_dict: Dict[object, List[str]],
    lam: float,
    w_max: float = 0.20,
    Rf: float = 0.045,
    maxiter: int = 300,
    max_passes: Optional[int] = None,
) -> Dict[str, object]:
    """
    Top-level entry point for the solver -- this is what quant_terminal.py
    should call (both the main Stage 4B solver AND the Tab 5 sandbox), NOT
    `run_true_two_stage_optimization` directly, so the singleton-split repair
    is always applied consistently in both places.

    Iterative Multi-Pass procedure:
      1. Run Stage1+Stage2 on the CURRENT cluster structure (starts as the
         input `clusters_dict`, as given).
      2. Log the full pass (clusters used, weights, g, feasibility, message)
         to `history`.
      3. Check `_identify_throttling_tickers` against this pass's own g*
         (and, if it converged, its solved weights too).
      4. If nothing is throttling AND the pass was feasible -> stop, this is
         the final answer.
         If the pass was infeasible but nothing new was found to promote ->
         this should not happen (see the necessity argument in
         `_identify_throttling_tickers`'s docstring: infeasibility implies
         some g_i > w_max exists) -- raises RuntimeError as an internal-logic
         guard rather than looping forever.
         If every currently-flagged ticker is ALREADY its own singleton (i.e.
         nothing new to split off) -> stop; this is as resolved as the
         structure can get.
      5. Otherwise, carve the newly-identified dominant ticker(s) into their
         own singleton clusters and repeat from step 1.

    Bounded by `max_passes` (default: total number of assets -- a safe upper
    bound, since at most every asset could end up promoted one at a time; in
    practice this resolves in 1-3 passes for realistic universes). Hitting
    `max_passes` without stabilizing raises ValueError rather than silently
    returning a still-throttled result.

    Returns
    -------
    dict with keys:
        "history"            : list of dicts, one per pass, each with:
                                  "pass"            : int, 1-indexed
                                  "clusters_dict"   : dict, structure used THIS pass (str keys)
                                  "weights"         : dict or None (None if this pass was infeasible)
                                  "g"               : dict, Stage 1 intra-cluster weights this pass
                                  "infeasible"      : bool
                                  "message"         : str or None
                                  "newly_promoted"  : list[str] or None -- tickers carved out AFTER this
                                                       pass to produce the NEXT pass's structure (None on
                                                       the final/stable pass, since nothing more was promoted)
        "final"              : dict -- the last (stable) pass's full Stage1+Stage2 output, the one that
                                should actually be used downstream.
        "split_triggered"    : bool -- True if more than one pass was needed.
        "promoted_tickers"   : list[str] -- ALL tickers promoted across every pass, in the order promoted.
        "final_clusters_dict": dict -- the cluster structure used in the final/stable pass.
        "n_passes"           : int
    """
    all_tickers = [t for members in clusters_dict.values() for t in members]
    if max_passes is None:
        max_passes = max(len(all_tickers), 1)

    current_clusters = {k: list(v) for k, v in clusters_dict.items()}
    history: List[Dict[str, object]] = []
    promoted_so_far: List[str] = []
    pass_num = 1

    while True:
        result, infeasible = _run_single_pass(mu_vec, sigma_eps, K_matrix, current_clusters, lam, w_max, Rf, maxiter)

        weights_dict = result["weights"].to_dict() if (not infeasible and result.get("weights") is not None) else None
        history.append({
            "pass": pass_num,
            "clusters_dict": {str(k): list(v) for k, v in current_clusters.items()},
            "weights": weights_dict,
            "g": result["g"].to_dict(),
            "infeasible": infeasible,
            "message": result.get("message"),
            "newly_promoted": None,
        })

        dominant = _identify_throttling_tickers(
            result["g"], current_clusters, w_max,
            cluster_W=(result.get("cluster_W") if not infeasible else None), infeasible=infeasible,
        )

        if not dominant:
            if infeasible:
                raise RuntimeError(
                    "run_optimization_with_singleton_split: pass reported infeasibility but no throttling "
                    "ticker (g_i > w_max) could be identified -- this should not happen; please report as a bug."
                )
            break  # stable, feasible, nothing throttling -> done

        new_dominant = [t for t in dominant if t not in promoted_so_far]
        if not new_dominant:
            # Every flagged ticker is ALREADY its own singleton this round -- structurally nothing
            # further can be carved out, so this is as resolved as it gets. (Also the loop's only
            # defense against infinite recursion: a singleton always has g_i=1.0 > w_max forever,
            # but `_identify_throttling_tickers` already skips single-member clusters entirely, so
            # in practice this branch is a secondary safety net, not the primary guard.)
            break

        if pass_num >= max_passes:
            raise ValueError(
                f"run_optimization_with_singleton_split: reached max_passes={max_passes} without finding a "
                f"stable cluster structure (still throttling: {new_dominant}). Raise w_max, or check whether "
                f"the fundamentals genuinely justify this much concentration."
            )

        history[-1]["newly_promoted"] = new_dominant
        current_clusters = _build_split_clusters(current_clusters, new_dominant)
        promoted_so_far.extend(new_dominant)
        pass_num += 1

    return {
        "history": history,
        "final": result,
        "split_triggered": len(promoted_so_far) > 0,
        "promoted_tickers": promoted_so_far,
        "final_clusters_dict": current_clusters,
        "n_passes": len(history),
    }


