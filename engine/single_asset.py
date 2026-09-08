"""
engine/single_asset.py
========================
Single-Asset Projection & Diagnostic Engine (Research module, Etap 4
follow-up), extended (2026-09-08 follow-up) into a Walk-Forward Calibration
& Backfit Engine. Pure math -- zero Dash/I/O.

Three layers, in order of when they were added:
1. `_compute_mu_h` / `compute_single_asset_projection` -- the forward-looking
   projection formula + probability cone for ONE asset (unchanged math from
   the first Etap 4 follow-up, just refactored so the pure mu_h calculation
   is its own function -- see note below).
2. `evaluate_historical_accuracy` -- walk-forward backtest: for each past
   fundamental snapshot where `horizon` trading SESSIONS have already
   elapsed (using the asset's own real trading calendar, not calendar
   days), compares what the model predicted back then against what
   actually happened.
3. `fit_single_asset_parameters` -- fits (alpha, gamma, kappa, eta) via
   scipy.optimize.minimize (L-BFGS-B, box-bounded) to minimize the mean
   squared return-prediction error across every evaluable historical entry.

Refactor note: `_compute_mu_h` was extracted out of what is now
`compute_single_asset_projection` specifically so `evaluate_historical_accuracy`
and `fit_single_asset_parameters` (which each evaluate mu_h many times --
once per historical entry, and for fitting, once per entry per optimizer
iteration) don't pay for building the full forward cone (t_days/p_proj/
p_upper/p_lower arrays, sigma/delta_down estimation, norm.cdf) on every
call -- that machinery is only needed once, for display, not for the
scalar mu_h a backtest/fit actually needs.

eta ("Execution / Realization Factor", confirmed design, 2026-09-07): a
manual multiplicative correction for a company's historical track record
of delivering vs. missing its own guidance / analyst consensus. Neutral
default 1.0. Now also a FITTABLE parameter (2026-09-08 follow-up) -- the
walk-forward backtest is exactly the "automatic realized-vs-expected
comparison" flagged as future work in the first pass.
"""

from __future__ import annotations

import bisect
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import least_squares

HORIZON_DAYS = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252}

# Domyslne zestawy parametrow per horyzont (confirmed, 2026-09-07 conversation).
# N_ref=15 wspoldzielone przez wszystkie horyzonty -- NIE jest czescia theta
# dopasowywanego przez fit_single_asset_parameters (potwierdzone, 2026-09-08:
# theta = (alpha, gamma, kappa, eta) tylko).
DEFAULT_HORIZON_PARAMS: Dict[str, Dict[str, float]] = {
    "1M": {"alpha": 0.15, "gamma": 0.20, "kappa": 1.60, "eta": 1.00, "n_ref": 15},
    "3M": {"alpha": 0.30, "gamma": 0.30, "kappa": 1.20, "eta": 1.00, "n_ref": 15},
    "6M": {"alpha": 0.50, "gamma": 0.40, "kappa": 0.80, "eta": 1.00, "n_ref": 15},
    "1Y": {"alpha": 0.70, "gamma": 0.45, "kappa": 0.40, "eta": 1.00, "n_ref": 15},
}

# Granice dopasowania (confirmed, 2026-09-08).
FIT_BOUNDS: Dict[str, tuple] = {"alpha": (0.0, 1.0), "gamma": (0.0, 1.5), "kappa": (0.0, 2.0), "eta": (0.5, 2.0)}

# Fitting 4 wolnych parametrow do 0-1 punktow danych jest matematycznie
# bezsensowne (skrajnie niedookreslone), nie tylko "mniej dokladne" --
# fit_single_asset_parameters odmawia ponizej tego progu zamiast zwracac
# fikcyjnie pewny wynik. Z DOKLADNIE tym progiem fit nadal jest zwracany,
# ale to nadal niedookreslony problem (4 parametry, 2 punkty) -- polityke
# "ostroznego traktowania malej proby" ustala UI (np. etykieta "n=2, wynik
# moze byc niestabilny"), nie ta funkcja.
MIN_ENTRIES_FOR_FIT = 2


def _resolve_params(horizon: str, params: Optional[Dict[str, float]]) -> Dict[str, float]:
    if horizon not in HORIZON_DAYS:
        raise ValueError(f"unknown horizon {horizon!r}, expected one of {list(HORIZON_DAYS)}")
    p = dict(DEFAULT_HORIZON_PARAMS[horizon])
    if params:
        p.update({k: v for k, v in params.items() if v is not None})
    return p


def _compute_mu_h(P0: float, Ti: float, Thigh: float, Tlow: float, Ni: float,
                   G_2Y: float, delta_eps_90d: float, horizon: str,
                   params: Optional[Dict[str, float]] = None) -> float:
    """
    Lean core formula (no cone, no diagnostics) -- see
    compute_single_asset_projection's docstring for the full formula
    writeup. Returns just mu_h. This is what evaluate_historical_accuracy /
    fit_single_asset_parameters call repeatedly; compute_single_asset_projection
    calls it exactly once per display render.
    """
    p = _resolve_params(horizon, params)
    alpha, gamma, kappa, eta, n_ref = p["alpha"], p["gamma"], p["kappa"], p["eta"], p["n_ref"]
    delta_t = HORIZON_DAYS[horizon]

    P0 = float(P0) if P0 else 0.0
    Ti = float(Ti) if Ti else 0.0
    Thigh = float(Thigh) if Thigh else 0.0
    Tlow = float(Tlow) if Tlow else 0.0
    Ni = float(Ni) if Ni else 0.0
    G_2Y = float(G_2Y) if G_2Y is not None else 0.0
    delta_eps_90d = float(delta_eps_90d) if delta_eps_90d is not None else 0.0

    S_i = (Thigh - Tlow) / Ti if Ti != 0 else 0.0
    U_adj = ((Ti - P0) / P0) * np.exp(-gamma * S_i) if P0 > 0 else 0.0
    G_h = (1.0 + G_2Y / 100.0) ** (delta_t / 504.0) - 1.0
    base_return = (1.0 - alpha) * U_adj + alpha * G_h
    M_rev = 1.0 + kappa * (delta_eps_90d / 100.0)
    A_N = (1.0 - np.exp(-Ni / n_ref)) if n_ref > 0 else 0.0
    return float(base_return * M_rev * A_N * eta)


def compute_single_asset_projection(
    P0: float, Ti: float, Thigh: float, Tlow: float, Ni: float,
    G_2Y: float, delta_eps_90d: float,
    historical_prices,
    horizon: str = "1Y",
    params: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """
    Section: Single-Asset Projection & Diagnostic Engine.

        S_i         = (T_high - T_low) / T_i
        U_adj       = ((T_i - P0) / P0) * exp(-gamma * S_i)
        G_h         = (1 + G_2Y/100)^(Delta_t / 504) - 1
        BaseReturn  = (1 - alpha) * U_adj + alpha * G_h
        M_rev       = 1 + kappa * (delta_eps_90d / 100)
        A_N         = 1 - exp(-N_i / N_ref)
        mu_h        = BaseReturn * M_rev * A_N * eta                      [see _compute_mu_h]

    Forward cone (from `historical_prices`, ideally 5Y daily closes for this
    ONE asset -- sigma_ann/delta_down_ann are annualized from its own daily
    log-return series):

        P_proj(t)  = P0 * (1 + mu_h * t/Delta_t)
        P_upper(t) = P_proj(t) * exp(+1.645 * sigma_ann      * sqrt(t/252))   [P90]
        P_lower(t) = P_proj(t) * exp(-1.645 * delta_down_ann * sqrt(t/252))   [P10]

    Diagnostics:
        P_profit         = norm.cdf( ln(1 + mu_h) / (sigma_ann * sqrt(Delta_t/252)) )
        Asymmetry_Ratio  = (P_upper[-1] - P0) / max(1e-4, P0 - P_lower[-1])

    Parameters / Returns: unchanged from the first Etap 4 follow-up -- see
    module docstring for the refactor note (mu_h now delegates to
    `_compute_mu_h`, no numeric change).
    """
    if horizon not in HORIZON_DAYS:
        raise ValueError(f"compute_single_asset_projection: unknown horizon {horizon!r}, expected one of {list(HORIZON_DAYS)}")

    delta_t = HORIZON_DAYS[horizon]
    p = _resolve_params(horizon, params)
    mu_h = _compute_mu_h(P0, Ti, Thigh, Tlow, Ni, G_2Y, delta_eps_90d, horizon, p)

    # Skladowe posrednie -- tylko do wgladu/diagnostyki w zwracanym dict,
    # przeliczone tu jeszcze raz (tanio, brak petli) zeby nie zmieniac
    # ksztaltu zwracanego slownika po refaktorze.
    P0f = float(P0) if P0 else 0.0
    Tif = float(Ti) if Ti else 0.0
    Thighf = float(Thigh) if Thigh else 0.0
    Tlowf = float(Tlow) if Tlow else 0.0
    S_i = (Thighf - Tlowf) / Tif if Tif != 0 else 0.0
    U_adj = ((Tif - P0f) / P0f) * np.exp(-p["gamma"] * S_i) if P0f > 0 else 0.0
    G_h = (1.0 + (float(G_2Y) if G_2Y is not None else 0.0) / 100.0) ** (delta_t / 504.0) - 1.0
    base_return = (1.0 - p["alpha"]) * U_adj + p["alpha"] * G_h
    M_rev = 1.0 + p["kappa"] * ((float(delta_eps_90d) if delta_eps_90d is not None else 0.0) / 100.0)
    A_N = (1.0 - np.exp(-(float(Ni) if Ni else 0.0) / p["n_ref"])) if p["n_ref"] > 0 else 0.0

    prices = pd.Series(historical_prices).dropna().astype(float)
    if len(prices) >= 30:
        rets = np.log(prices / prices.shift(1)).dropna()
        sigma_ann = float(rets.std() * np.sqrt(252))
        delta_down_ann = float(np.sqrt(np.mean(np.minimum(rets.values, 0.0) ** 2)) * np.sqrt(252))
    else:
        sigma_ann, delta_down_ann = 0.0, 0.0
    sigma_ann_safe = max(sigma_ann, 1e-6)
    delta_down_safe = max(delta_down_ann, 1e-6)

    t_days = np.arange(0, delta_t + 1)
    frac = t_days / delta_t if delta_t > 0 else np.zeros_like(t_days, dtype=float)
    p_proj = P0f * (1.0 + mu_h * frac)
    sqrt_t = np.sqrt(t_days / 252.0)
    p_upper = p_proj * np.exp(1.645 * sigma_ann_safe * sqrt_t)
    p_lower = p_proj * np.exp(-1.645 * delta_down_safe * sqrt_t)

    denom = sigma_ann_safe * np.sqrt(delta_t / 252.0)
    p_profit = float(norm.cdf(np.log(1.0 + mu_h) / denom)) if mu_h > -1.0 else 0.0
    asymmetry_ratio = float((p_upper[-1] - P0f) / max(1e-4, (P0f - p_lower[-1])))

    return {
        "horizon": horizon, "delta_t": delta_t,
        "mu_h": float(mu_h), "base_return": float(base_return), "u_adj": float(U_adj), "g_h": float(G_h),
        "m_rev": float(M_rev), "a_n": float(A_N),
        "sigma_ann": sigma_ann, "delta_down_ann": delta_down_ann,
        "t_days": t_days, "p_proj": p_proj, "p_upper": p_upper, "p_lower": p_lower,
        "p_profit": p_profit, "asymmetry_ratio": asymmetry_ratio,
        "params_used": {"alpha": p["alpha"], "gamma": p["gamma"], "kappa": p["kappa"], "eta": p["eta"], "n_ref": p["n_ref"]},
    }


# ---------------------------------------------------------------------------
# Walk-Forward Calibration & Backfit Engine (2026-09-08 follow-up)
# ---------------------------------------------------------------------------

def _target_date_after_sessions(sorted_dates: List[str], from_date_str: str, n_sessions: int) -> Optional[str]:
    """
    Advances `from_date_str` by `n_sessions` TRADING sessions (not calendar
    days), using `sorted_dates` (the asset's own real trading calendar --
    its 5Y price series' dates, sorted ascending) as the source of truth.
    Confirmed design: horizons are defined in trading sessions, so a naive
    calendar-day shift would systematically misalign the evaluation date
    around weekends/holidays.

    Returns None if `from_date_str` is beyond the available calendar, or if
    advancing `n_sessions` from it runs past the end of the available
    calendar (the horizon hasn't elapsed yet within the data we have --
    NOT an error, just "not evaluable yet").
    """
    if not sorted_dates:
        return None
    pos = bisect.bisect_left(sorted_dates, from_date_str)
    if pos >= len(sorted_dates):
        return None
    target_pos = pos + n_sessions
    if target_pos >= len(sorted_dates):
        return None
    return sorted_dates[target_pos]


def _collect_evaluable(history_entries: List[dict], price_series: "pd.Series", horizon: str):
    """Shared filtering logic between evaluate_historical_accuracy and
    fit_single_asset_parameters: which historical entries have both (a) a
    fully-elapsed horizon within the available price calendar, and (b) a
    real, non-NaN close price at the resulting target date. Returns a list
    of (entry, target_date, p_real, r_real) tuples, chronological."""
    if price_series is None or len(price_series) == 0:
        return []
    sorted_dates = sorted(str(d) for d in price_series.index)
    delta_t = HORIZON_DAYS[horizon]

    out = []
    for entry in history_entries:
        t_k = entry.get("Date")
        if not t_k:
            continue
        target_date = _target_date_after_sessions(sorted_dates, t_k, delta_t)
        if target_date is None:
            continue
        p0_k = entry.get("P0")
        if not p0_k or p0_k <= 0:
            continue
        try:
            p_real = float(price_series.loc[target_date])
        except KeyError:
            continue
        if pd.isna(p_real):
            continue
        r_real = (p_real - p0_k) / p0_k
        out.append((entry, target_date, p_real, r_real))
    return out


def evaluate_historical_accuracy(history_entries: List[dict], price_series: "pd.Series",
                                  horizon: str = "1Y", params: Optional[Dict[str, float]] = None) -> List[Dict[str, object]]:
    """
    Walk-forward backtest: for each historical fundamental snapshot in
    `history_entries` (data/company_store.py's per-ticker entry list) where
    `horizon` trading sessions have already elapsed (per the asset's own
    real trading calendar in `price_series`), compares the model's
    predicted return AT THAT TIME against what actually happened.

    Entries where the horizon hasn't elapsed yet within the available price
    history are silently skipped (not included with null placeholders) --
    they simply aren't evaluable backtest points yet.

    Parameters
    ----------
    history_entries : list of dict
        Exactly the shape returned by data.company_store.get_history() --
        this function does not import that module (one-directional
        dependency rule: engine/ never imports data/), the caller passes
        the entries in.
    price_series : pd.Series
        Index = date strings ("YYYY-MM-DD"), values = daily close prices
        for this one asset (ideally the same 5Y cache used elsewhere).
    horizon : str
    params : dict or None
        Same (alpha, gamma, kappa, eta, n_ref) override semantics as
        compute_single_asset_projection.

    Returns
    -------
    List[dict], chronological by date_k, each with keys:
        "date_k", "p0_k", "mu_model", "p_pred", "date_target", "p_real",
        "r_real", "error", "price_error_pct", "realization_ratio"
    "error" = mu_model - r_real (positive => model overshot / too optimistic;
    negative => model undershot / reality beat the model).
    "realization_ratio" = r_real / mu_model, or None if mu_model is ~0
    (a ratio against a near-zero denominator is not meaningful, not "very
    large" -- reported as unavailable rather than as a misleading extreme).
    """
    evaluable = _collect_evaluable(history_entries, price_series, horizon)
    results = []
    for entry, target_date, p_real, r_real in evaluable:
        mu_model = _compute_mu_h(
            P0=entry.get("P0"), Ti=entry.get("Target_Consensus"), Thigh=entry.get("Target_High"),
            Tlow=entry.get("Target_Low"), Ni=entry.get("N_analysts"), G_2Y=entry.get("EPS_CAGR"),
            delta_eps_90d=entry.get("EPS_Rev_90d"), horizon=horizon, params=params,
        )
        p0_k = entry["P0"]
        p_pred = p0_k * (1.0 + mu_model)
        error = mu_model - r_real
        price_error_pct = (p_pred - p_real) / p_real * 100.0 if p_real else None
        realization_ratio = (r_real / mu_model) if abs(mu_model) > 1e-6 else None

        results.append({
            "date_k": entry["Date"], "p0_k": p0_k, "mu_model": mu_model, "p_pred": p_pred,
            "date_target": target_date, "p_real": p_real, "r_real": r_real,
            "error": error, "price_error_pct": price_error_pct, "realization_ratio": realization_ratio,
        })
    return results


STABILITY_LABELS = [(0.15, "Stabilne"), (0.40, "Umiarkowanie stabilne"), (1.0, "Niestabilne")]
STABILITY_FALLBACK_LABEL = "Bardzo niestabilne / niezidentyfikowane"


def stability_label(cv: Optional[float]) -> str:
    """Coefficient-of-variation -> a plain-language stability tier. `cv=None`
    (point estimate ~0, so a relative CV is undefined) maps to the most
    cautious label rather than silently omitting a verdict."""
    if cv is None:
        return STABILITY_FALLBACK_LABEL
    for threshold, label in STABILITY_LABELS:
        if cv < threshold:
            return label
    return STABILITY_FALLBACK_LABEL


def fit_single_asset_parameters(history_entries: List[dict], price_series: "pd.Series",
                                 horizon: str = "1Y") -> Optional[Dict[str, object]]:
    """
    Fits (alpha, gamma, kappa, eta) to minimize the sum of squared
    return-prediction residuals (mu_model - r_real) across every historical
    entry where `horizon` has already elapsed. N_ref stays FIXED at
    DEFAULT_HORIZON_PARAMS[horizon]["n_ref"] -- confirmed not part of theta.

    Uses `scipy.optimize.least_squares` (Trust Region Reflective, box-bounded
    per FIT_BOUNDS), NOT `scipy.optimize.minimize` on a hand-rolled scalar
    MSE objective -- confirmed correction after a real finding during
    testing (2026-09-08): on an exact, zero-noise synthetic recovery test
    with only 8 data points, `minimize(..., method="L-BFGS-B")` on the
    scalar-SSE objective converged to a materially WRONG parameter set
    (kappa off by 0.5) despite reporting a near-perfect MSE -- an optimizer
    robustness failure (getting stuck away from the true optimum), not
    (only) a fundamental non-identifiability of the model. Switching to
    `least_squares`, which is purpose-built for exactly this
    sum-of-squared-residuals problem structure rather than treating it as a
    generic scalar function, recovered the exact true parameters on the
    same test case. This also yields the Jacobian at the solution for free,
    which is what the stability diagnostics below are built on.

    Stability diagnostics (asymptotic, from the Jacobian at the solution):
    the classical nonlinear-least-squares approximation
        Cov(theta) ~= sigma^2 * (J^T J)^-1,   sigma^2 = SSE / (n_obs - n_params)
    gives a standard error per parameter. Chosen over bootstrap resampling
    (tried first, then abandoned) after a real, documented finding: with a
    small, ZERO-NOISE synthetic dataset that had a genuine structural
    degeneracy (several different theta combinations fit those exact 8
    points almost equally well), bootstrap resampling of THAT SAME fixed
    set of 8 points reported "Stabilne" for every parameter -- resampling
    only reweights the SAME underlying feature combinations, so it cannot
    reveal a degeneracy that is a property of the feature diversity itself,
    not of sampling noise. Re-tested with REALISTIC noise added (see
    PROJECT_CONTEXT.md): the Jacobian-based standard error correctly
    reported enormous uncertainty for kappa (e.g. 0.000 +/- 6.121 on 8
    noisy points) and visibly tighter (but still often wide) intervals with
    30 points -- behaving exactly as a stability diagnostic should.
    `condition_number` (of J^T J) is also returned: very high values flag
    that the fit is in or near a flat/degenerate direction even before
    looking at any individual parameter's stderr.

    This is still an asymptotic APPROXIMATION (exact only as n_obs -> infinity),
    not a guarantee -- but it is the standard, well-founded tool for this
    exact problem class, and it behaved correctly in every test case run
    against it, unlike the bootstrap alternative.

    Returns None if fewer than MIN_ENTRIES_FOR_FIT evaluable entries exist.
    On success, returns:
        {"alpha", "gamma", "kappa", "eta"}          -- point estimates
        "n_evaluable", "mse"                         -- same as before
        "condition_number"                           -- of J^T J at the solution
        "stability": {
            "alpha": {"stderr": float, "cv": float|None, "label": str},
            "gamma": {...}, "kappa": {...}, "eta": {...},
        }
    `cv` (coefficient of variation) = stderr / abs(point_estimate); `None`
    when the point estimate is ~0 (a relative measure is undefined against
    a near-zero denominator, not "perfectly stable").
    """
    evaluable = _collect_evaluable(history_entries, price_series, horizon)
    if len(evaluable) < MIN_ENTRIES_FOR_FIT:
        return None

    n_ref = DEFAULT_HORIZON_PARAMS[horizon]["n_ref"]

    def residuals(theta):
        alpha, gamma, kappa, eta = theta
        p = {"alpha": alpha, "gamma": gamma, "kappa": kappa, "eta": eta, "n_ref": n_ref}
        return np.array([
            _compute_mu_h(entry.get("P0"), entry.get("Target_Consensus"), entry.get("Target_High"),
                           entry.get("Target_Low"), entry.get("N_analysts"), entry.get("EPS_CAGR"),
                           entry.get("EPS_Rev_90d"), horizon, p) - r_real
            for entry, _td, _pr, r_real in evaluable
        ])

    defaults = DEFAULT_HORIZON_PARAMS[horizon]
    x0 = [defaults["alpha"], defaults["gamma"], defaults["kappa"], defaults["eta"]]
    lower = [FIT_BOUNDS[k][0] for k in ["alpha", "gamma", "kappa", "eta"]]
    upper = [FIT_BOUNDS[k][1] for k in ["alpha", "gamma", "kappa", "eta"]]

    result = least_squares(residuals, x0, bounds=(lower, upper))
    alpha, gamma, kappa, eta = (float(v) for v in result.x)
    n_obs = len(evaluable)
    sse = float(np.sum(result.fun ** 2))
    mse = sse / n_obs

    J = result.jac
    dof = max(1, n_obs - 4)
    residual_var = sse / dof
    JTJ = J.T @ J
    try:
        condition_number = float(np.linalg.cond(JTJ))
        cov = residual_var * np.linalg.inv(JTJ)
        stderrs = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        condition_number = float("inf")
        stderrs = [float("nan")] * 4

    point = {"alpha": alpha, "gamma": gamma, "kappa": kappa, "eta": eta}
    stability = {}
    for name, se in zip(["alpha", "gamma", "kappa", "eta"], stderrs):
        se = float(se)
        val = point[name]
        cv = (se / abs(val)) if abs(val) > 1e-9 and np.isfinite(se) else None
        stability[name] = {"stderr": se, "cv": cv, "label": stability_label(cv)}

    return {
        "alpha": alpha, "gamma": gamma, "kappa": kappa, "eta": eta,
        "n_evaluable": n_obs, "mse": mse, "condition_number": condition_number,
        "stability": stability,
    }