"""
engine/evaluation.py
=====================
Historical / forward portfolio backtest evaluation: equity curve
reconstruction and empirical tail-risk / TPS metrics for a fixed weight
vector.

No Dash / UI / disk-I/O code lives here on purpose. Moved out of the old
monolithic tps_solver.py (Etap 0 architecture split, PROJECT_CONTEXT.md)
with NO behavior change -- verbatim relocation.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from engine.risk import TRADING_DAYS_PER_YEAR

EPSILON = 1e-6  # safety epsilon added to TPS denominators to avoid division by zero


def evaluate_portfolio_performance(
    weights: pd.Series,
    returns_df: pd.DataFrame,
    is_log_returns: bool = True,
    V0: float = 100.0,
    quantile: float = 0.10,
    downside_cov_matrix: Optional[pd.DataFrame] = None,
    P_vec: Optional[pd.Series] = None,
    Rf: float = 0.045,
) -> Dict[str, object]:
    """
    Reconstructs the historical portfolio equity curve for a fixed weight
    vector and evaluates its empirical tail-risk / TPS metrics.

    IMPORTANT: equity-curve compounding (V_t = V_0 * cumprod(1 + R_t)) is only
    valid for SIMPLE returns. If `returns_df` holds log returns (the default
    assumption, `is_log_returns=True`), they are converted via
    `simple = exp(log) - 1` BEFORE the weighted sum / compounding. This is a
    deliberate correction -- do not "simplify" this back to using log returns
    directly in the cumprod, that would silently misstate the equity curve.

    Parameters
    ----------
    weights : pd.Series
        Final portfolio weights per ticker (e.g. from `optimize_portfolio_slsqp`,
        or a 1/N equal-weight Series for a benchmark comparison).
    returns_df : pd.DataFrame
        Daily returns, one column per ticker, aligned on a common date index.
        Only the columns in `weights.index` (intersected with `returns_df.columns`)
        are used.
    is_log_returns : bool, default True
        Whether `returns_df` holds log returns (converted internally to simple
        returns for compounding) or already-simple returns (used as-is).
    V0 : float, default 100.0
        Starting notional value of the synthetic equity curve.
    quantile : float, default 0.10
        Empirical quantile used for the tail-drawdown floor (CDD_quantile).
    downside_cov_matrix, P_vec, Rf : optional
        If BOTH `downside_cov_matrix` and `P_vec` are supplied, this function
        also computes a backtest-consistent portfolio TPS using the realized
        annualized return from this same equity curve (distinct from the
        forward-looking mu_P used inside the optimizer -- this is a trailing,
        realized-history cross-check, not the optimization objective).

    Returns
    -------
    dict with keys:
        "dates"              : list[str] (ISO date strings), the return series' date index
        "portfolio_returns"  : list[float], daily SIMPLE portfolio returns
        "equity_curve"       : list[float], V_t series starting at V0
        "drawdown_series"    : list[float], DD_t (<=0, e.g. -0.10 for -10%)
        "cdd_quantile"       : float, -Quantile_q(DD_t)  (positive magnitude, e.g. 0.185 for 18.5%)
        "realized_return_ann": float, annualized mean simple daily return * 252 (trailing/backtest figure)
        "delta_ann"          : float or None, sqrt(w . Sigma_downside . w) if downside_cov_matrix given
        "tps_backtest"       : float or None, (realized_return_ann - Rf) / (delta_ann * P_p + eps),
                                only computed if both downside_cov_matrix and P_vec are given
    """
    common = [t for t in weights.index if t in returns_df.columns]
    if not common:
        raise ValueError("evaluate_portfolio_performance: no overlap between weights.index and returns_df.columns.")

    w = weights.reindex(common)
    w = w / w.sum() if w.sum() > 1e-12 else pd.Series(1.0 / len(common), index=common)

    r = returns_df[common].dropna(how="all")
    if is_log_returns:
        simple_r = np.exp(r) - 1.0
    else:
        simple_r = r

    portfolio_returns = (simple_r.fillna(0.0) @ w.values).astype(float)

    equity_curve = V0 * (1.0 + portfolio_returns).cumprod()
    running_max = equity_curve.cummax()
    drawdown_series = (equity_curve - running_max) / running_max

    cdd_quantile = float(-drawdown_series.quantile(quantile))
    realized_return_ann = float(portfolio_returns.mean() * TRADING_DAYS_PER_YEAR)

    delta_ann = None
    tps_backtest = None
    if downside_cov_matrix is not None and P_vec is not None:
        cov_arr = downside_cov_matrix.loc[common, common].values
        delta_ann = float(np.sqrt(max(w.values @ cov_arr @ w.values, 0.0)))
        p_p = float((w * P_vec.reindex(common)).sum())
        tps_backtest = (realized_return_ann - Rf) / (delta_ann * p_p + EPSILON)

    return {
        "dates": [str(d) for d in portfolio_returns.index],
        "portfolio_returns": portfolio_returns.tolist(),
        "equity_curve": equity_curve.tolist(),
        "drawdown_series": drawdown_series.tolist(),
        "cdd_quantile": cdd_quantile,
        "realized_return_ann": realized_return_ann,
        "delta_ann": delta_ann,
        "tps_backtest": tps_backtest,
    }
