"""
engine/risk.py
==============
Pure risk math: regularized Estrada downside semi-covariance (Sigma_eps),
drawdown series, CDD quantile floor, and the Discrete Crash-Overlap Matrix K.

No Dash / UI / disk-I/O code lives here on purpose -- this module is pure
Pandas/NumPy so it can be unit-tested and imported independently of the app.
Moved out of the old monolithic tps_solver.py (Etap 0 architecture split,
PROJECT_CONTEXT.md) with NO behavior change -- every function below is a
verbatim relocation, not a rewrite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RIDGE_EPSILON = 1e-8  # Tikhonov ridge added to Sigma_downside's diagonal (Section 3.2) --
                       # kept separate from optimizer.EPSILON: that one guards a scalar
                       # division, this one guards a matrix against near-singularity.
TRADING_DAYS_PER_YEAR = 252


def compute_estrada_matrix(returns_df: pd.DataFrame, annualize: bool = True, apply_ridge: bool = True) -> pd.DataFrame:
    """
    Estrada-style downside semi-covariance matrix, thresholded at a Minimum
    Acceptable Return (MAR) of 0 -- NOT at each asset's own mean. This is a
    distinct, simpler construction than a mean-demeaned semi-covariance: it
    only penalizes literal negative-return days, which is what Section 4.5
    calls for.

        Sigma_downside[i, j] = (1/T) * sum_t( min(r_i,t, 0) * min(r_j,t, 0) ) * 252

    Tikhonov ridge regularization (Section 3.2):
        Sigma_eps = Sigma_downside + RIDGE_EPSILON * I_N,  RIDGE_EPSILON = 1e-8

    Note: Sigma_downside = R_neg.T @ R_neg / T is a Gram matrix, so it is
    already PSD by construction -- the ridge term isn't needed to "make it
    PSD". Its actual job is guarding against near-singularity / ill-conditioning
    (e.g. N approaching T, or near-collinear return series among two names in
    the same cluster), which is what would otherwise destabilize the SLSQP
    solvers that consume this matrix inside sqrt(w^T @ Sigma_eps @ w).

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily returns, one column per ticker (log returns are fine here --
        this is only ever used inside a quadratic form, see module docstring).
    annualize : bool, default True
        If True, multiplies the raw daily matrix by TRADING_DAYS_PER_YEAR (252).
        Set False if you want the raw daily-frequency matrix instead.
    apply_ridge : bool, default True
        If True (default), adds RIDGE_EPSILON * I_N to the (annualized) matrix
        before returning. Exposed as a flag mainly for testing/inspection of
        the raw matrix; production call sites should leave this at default.

    Returns
    -------
    pd.DataFrame
        N x N symmetric downside semi-covariance matrix, indexed/columned by
        the tickers in `returns_df.columns`. Diagonal entries are each asset's
        own downside semi-variance (plus RIDGE_EPSILON if apply_ridge=True);
        sqrt(diagonal) gives the annualized downside semi-deviation used
        elsewhere as `semideviation_ann_vec`.
    """
    if returns_df.empty:
        raise ValueError("compute_estrada_matrix: returns_df is empty.")

    R = returns_df.values.astype(float)
    R_neg = np.minimum(R, 0.0)  # threshold at MAR = 0, NOT demeaned
    T, N = R.shape
    if T == 0:
        raise ValueError("compute_estrada_matrix: no observations (T=0).")

    sigma = (R_neg.T @ R_neg) / T
    if annualize:
        sigma = sigma * TRADING_DAYS_PER_YEAR

    if apply_ridge:
        sigma = sigma + RIDGE_EPSILON * np.eye(N)

    return pd.DataFrame(sigma, index=returns_df.columns, columns=returns_df.columns)


def semideviation_ann_from_matrix(downside_cov_matrix: pd.DataFrame) -> pd.Series:
    """
    Convenience accessor: annualized downside semi-deviation per asset is just
    sqrt of the diagonal of an already-annualized Estrada matrix. Kept as a
    tiny standalone helper so callers don't have to remember `np.sqrt(np.diag(...))`.
    """
    diag = np.diag(downside_cov_matrix.values)
    return pd.Series(np.sqrt(np.clip(diag, 0.0, None)), index=downside_cov_matrix.index)


# ---------------------------------------------------------------------------
# 1B. DRAWDOWN SERIES, CDD QUANTILE & DISCRETE CRASH-OVERLAP MATRIX (Section 3.3)
# ---------------------------------------------------------------------------
# Moved here from quant_terminal.py (was duplicated inline inside
# compute_tail_risk_metrics) so all pure risk-matrix math lives in this module,
# per the Section 5 module split: tps_solver.py = math, quant_terminal.py =
# UI/state/orchestration. quant_terminal.py now calls these instead of
# recomputing drawdowns itself.

def compute_drawdown_series(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    DD_i,t = (P_i,t - cummax(P_i,t)) / cummax(P_i,t)   [Section 3.3.A]

    Computed independently per column on that column's OWN full price history
    (not a shared/truncated universe window) so an asset's running maximum is
    never artificially reset by another asset's missing dates. `prices_df` is
    expected to be a raw close-price DataFrame with one column per ticker,
    aligned on a common calendar `DatetimeIndex` but very likely containing
    NaN gaps -- e.g. this project mixes NYSE/NASDAQ tickers with KRX
    (`005930.KS`), ASX (`LYC.AX`, `SIG.AX`) and LSE (`ANTO.L`) names, which
    all have different trading-holiday calendars, so ragged NaN gaps across
    columns are the normal case here, not an edge case.

    pandas' `.cummax()` correctly skips NaN gaps without corrupting the
    running max for later valid days (a NaN position stays NaN, but the next
    valid value still compares against the pre-gap running max) -- verified
    behavior, not an assumption -- so no `.dropna()` truncation is needed
    before calling this.

    Returns
    -------
    pd.DataFrame
        Same shape/index/columns as `prices_df`. NaN wherever the input price
        was NaN (no trading that day for that ticker) or before that ticker's
        first valid price.
    """
    dd = pd.DataFrame(index=prices_df.index, columns=prices_df.columns, dtype=float)
    for t in prices_df.columns:
        p = prices_df[t]
        running_max = p.cummax()
        dd[t] = (p - running_max) / running_max
    return dd


def compute_cdd_quantile_vec(dd_df: pd.DataFrame, quantile: float = 0.10) -> pd.Series:
    """
    CDD_q,i = -Quantile_q({DD_i,t})   [Section 3.3.A / empirical tail-drawdown floor]

    `pd.DataFrame.quantile` ignores NaN per column by default, so ragged
    histories (see `compute_drawdown_series`) are handled correctly without
    any extra alignment step here.
    """
    return (-dd_df.quantile(quantile, axis=0)).astype(float)


def compute_crash_overlap_matrix(prices_df: pd.DataFrame, quantile: float = 0.10) -> dict:
    """
    Section 3.3: Discrete Crash-Overlap Matrix K = J (.) S.

    Design decision -- PAIRWISE date intersection (confirmed with project owner,
    "Option A"): this project's universe routinely mixes tickers from different
    exchanges/calendars (see `compute_drawdown_series` docstring), so summing
    the Jaccard numerator/denominator over a single shared "T" for the whole
    universe would either (a) silently miscount days a ticker wasn't trading
    as "not in crash" (deflates J for any pair involving a foreign-exchange or
    newly-listed name), or (b) require truncating every asset's history down
    to the youngest listing in the universe, which is exactly the kind of
    look-ahead-adjacent data loss Section 4's T_min floor is trying to avoid.
    Instead, each pair (i, j) is scored only over the trading days where BOTH
    assets have a real (non-NaN) price -- i.e. per-pair intersection, not a
    universe-wide one. This is an O(N^2) loop over ticker pairs rather than a
    single vectorized matrix op; at this project's scale (single/low-double
    digit ticker counts) that is negligible (well under a second even at 5
    years of daily data) and keeps the crash-day accounting fully auditable.

    Steps
    -----
    A. DD_i,t                    -- `compute_drawdown_series`
    B. CDD_0.10,i                -- `compute_cdd_quantile_vec`
    C. Crash mask I_i,t = (DD_i,t <= -CDD_0.10,i)  (NaN days evaluate to False
       automatically under pandas' NaN-comparison semantics, so they never
       register as "crashing" -- but see the pairwise-intersection note above
       for why that alone isn't sufficient for J.)
    D. J_i,j = |I_i,t AND I_j,t| / |I_i,t OR I_j,t|, summed only over days
       where BOTH i and j have valid prices. J_i,i := 1.0 by convention.
    E. S = v @ v.T where v = CDD_0.10 vector (outer product, Section 3.3.B).
    F. K = J (.) S  (elementwise / Hadamard product, Section 3.3.C).

    Parameters
    ----------
    prices_df : pd.DataFrame
        Raw close prices, one column per ticker, DatetimeIndex. NaN gaps for
        non-trading days on a given exchange are expected and handled.
    quantile : float, default 0.10
        Empirical quantile used for the CDD tail-drawdown floor (matches
        `evaluate_portfolio_performance`'s `quantile` param elsewhere in this
        module -- keep these in sync if the project-wide convention changes).

    Returns
    -------
    dict with keys:
        "dd"          : pd.DataFrame, DD_i,t (Section 3.3.A), NaN-preserving.
        "cdd_vec"     : pd.Series, CDD_0.10,i per ticker (positive magnitude).
        "crash_mask"  : pd.DataFrame, boolean I_i,t (Section 3.3.A indicator).
        "J"           : pd.DataFrame, temporal crash-overlap matrix.
        "S"           : pd.DataFrame, severity outer-product matrix.
        "K"           : pd.DataFrame, composite crash-risk matrix K = J (.) S.
    All DataFrames/Series share the same ticker index/columns, in the order
    of `prices_df.columns`.
    """
    if prices_df.empty:
        raise ValueError("compute_crash_overlap_matrix: prices_df is empty.")

    tickers = list(prices_df.columns)
    N = len(tickers)

    dd_df = compute_drawdown_series(prices_df)
    cdd_vec = compute_cdd_quantile_vec(dd_df, quantile)

    # Crash indicator I_{i,t}. NaN <= x evaluates to False under pandas/numpy
    # comparison semantics, so missing-data days are never flagged as crashes.
    crash_mask = pd.DataFrame(index=dd_df.index, columns=tickers, dtype=bool)
    for t in tickers:
        crash_mask[t] = dd_df[t] <= -cdd_vec[t]

    valid_price = prices_df.notna()  # per-ticker "did this asset actually trade today"

    J = pd.DataFrame(np.eye(N), index=tickers, columns=tickers, dtype=float)
    for i in range(N):
        ti = tickers[i]
        for j in range(i + 1, N):
            tj = tickers[j]
            common = valid_price[ti] & valid_price[tj]  # Option A: pairwise date intersection
            if not common.any():
                j_val = 0.0
            else:
                Ii = crash_mask[ti] & common
                Ij = crash_mask[tj] & common
                union = int((Ii | Ij).sum())
                inter = int((Ii & Ij).sum())
                j_val = float(inter / union) if union > 0 else 0.0
            J.loc[ti, tj] = j_val
            J.loc[tj, ti] = j_val

    v = cdd_vec.reindex(tickers).values.reshape(-1, 1)
    S = pd.DataFrame(v @ v.T, index=tickers, columns=tickers)

    K = J * S  # Hadamard product -- element-wise, NOT matrix multiplication

    return {"dd": dd_df, "cdd_vec": cdd_vec, "crash_mask": crash_mask, "J": J, "S": S, "K": K}


