"""
engine/pairs.py
=================
Cointegration Pair Screener (2026-09-08 follow-up). Pure math -- zero Dash/I/O.

Scans a universe of tickers (the ~40 that already passed the project's
existing clustering/screening pipeline) for pairs whose prices move together
closely enough, in a statistically rigorous sense, to be worth a SOFT
relative-value nudge to their expected returns -- NOT a classical
market-neutral pairs-trading strategy. Confirmed design (2026-09-08
conversation): this project is Long-Only, rebalances on a fixed ~21-trading-day
(monthly) cadence, and a "wrong" pair costs at most one weaker month on one
hyper-growth name, not a leveraged blowup -- there is no daily execution
loop here, no short leg, and the eventual use of a qualifying pair's Z-score
(a bounded tanh-based adjustment to mu_i, implemented separately once this
screener is confirmed correct) is evaluated once per rebalance cycle, not
continuously.

Rewritten from scratch after auditing a third-party (Gemini-authored)
reference implementation that had three real, confirmed bugs, all fixed here:
1. The stated "p-value < 0.05" gate was never actually applied in the
   reference code's filtering loop -- only the half-life bound was checked,
   so pairs with p-value as high as 0.0958 (MU vs WDC) ended up in its
   "top 20" ranking despite failing the very criterion the report described.
2. No sanity check on the hedge ratio's sign/magnitude -- a negative hedge
   ratio (observed: GRAB vs MCHP, gamma=-0.608) means the two series move in
   OPPOSITE directions, which is not economically a "pair" for this
   same-direction relative-value mechanism at all, most likely a spurious
   regression that happened to pass the other filters.
3. Plain `statsmodels.tsa.stattools.adfuller()` applied directly to OLS
   regression residuals uses ADF's standard critical values, which assume a
   RAW observed series -- but residuals from an estimated 2-parameter
   regression have a different (non-standard) null distribution, so this
   systematically overstates evidence for cointegration (p-values read as
   more significant than they really are). Fixed by using
   `statsmodels.tsa.stattools.coint()`, which applies the correct
   MacKinnon-adjusted critical values for exactly this two-step
   Engle-Granger setup.

No Dash / UI / disk-I/O code lives here on purpose (one-directional
dependency rule: engine/ never imports data/ or ui/). Callers pass in an
already-fetched price DataFrame (e.g. via data.market_data.fetch_universe_prices).
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

TRADING_DAYS_2Y = 504  # 2 * 252, matches the drift-gate window described in the report

# Domyslne progi (confirmed design, 2026-09-08). Half-life zakres 5-63 sesji
# pochodzi wprost z raportu -- zostaje bez zmian na razie (miesieczny rytm
# rebalansu tego projektu moze docelowo uzasadniac zawezenie/rozszerzenie
# tego zakresu, ale to swiadoma, OSOBNA decyzja do podjecia PO potwierdzeniu,
# ze sam skaner liczy poprawnie -- nie zmieniane tutaj po cichu).
DEFAULT_MAX_DRIFT_PP = 35.0
DEFAULT_P_VALUE_MAX = 0.05
DEFAULT_HALF_LIFE_MIN = 5
DEFAULT_HALF_LIFE_MAX = 63
DEFAULT_MIN_HEDGE_RATIO = 0.3   # odrzuca hedge ratio bliskie 0 lub ujemne
DEFAULT_MAX_HEDGE_RATIO = 3.0   # odrzuca hedge ratio absurdalnie duze (skale cen niewspolmierne)


def _hedge_ratio_and_spread(log_price_a: pd.Series, log_price_b: pd.Series) -> Dict[str, object]:
    """
    OLS: log(P_A) = alpha + gamma * log(P_B) + epsilon_t.

    Returns {"alpha", "gamma", "spread"} -- `spread` is the residual series
    epsilon_t (same index as the inputs), used downstream both for the
    half-life estimate here and for the live Z-score at signal-evaluation
    time (not computed in this module -- this module only screens/ranks
    candidate pairs).
    """
    X = np.column_stack([np.ones(len(log_price_b)), log_price_b.values])
    y = log_price_a.values
    coeffs, _residuals, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    alpha, gamma = float(coeffs[0]), float(coeffs[1])
    spread = log_price_a - (alpha + gamma * log_price_b)
    return {"alpha": alpha, "gamma": gamma, "spread": spread}


def _half_life(spread: pd.Series) -> float:
    """
    Half-life of mean reversion (in trading sessions) from an AR(1) fit on
    the spread: Delta(spread_t) = c + rho * spread_{t-1} + noise.

        t_1/2 = -ln(2) / ln(1 + rho)

    Returns float("inf") if rho >= 0 (no mean reversion at all -- the
    spread's own AR(1) coefficient implies it does not decay back toward
    its mean, regardless of what the cointegration test's p-value says) or
    if rho <= -1 (degenerate/non-sensical, would make ln(1+rho) undefined).
    """
    lagged = spread.shift(1).dropna()
    delta = (spread - spread.shift(1)).dropna()
    lagged, delta = lagged.align(delta, join="inner")
    if len(lagged) < 10:
        return float("inf")

    X = np.column_stack([np.ones(len(lagged)), lagged.values])
    coeffs, _residuals, _rank, _sv = np.linalg.lstsq(X, delta.values, rcond=None)
    rho = float(coeffs[1])

    if rho >= 0 or rho <= -1 or abs(np.log1p(rho)) < 1e-10:
        return float("inf")
    return float(-np.log(2) / np.log1p(rho))


def test_pair_cointegration(prices_a: pd.Series, prices_b: pd.Series) -> Dict[str, object]:
    """
    Full Engle-Granger two-step test for ONE pair, on whatever price window
    the caller passes in (see find_cointegrated_pairs for the project's
    default 5Y window, matching data.market_data's existing convention).

    Returns
    -------
    dict with keys:
        "p_value"      : float, from statsmodels.tsa.stattools.coint (MacKinnon-adjusted,
                          correct for testing cointegration between two I(1) series --
                          NOT a plain adfuller() p-value on raw residuals, see module docstring).
        "hedge_ratio"  : float (gamma from the OLS regression)
        "alpha"        : float (OLS intercept)
        "half_life"    : float, trading sessions (float("inf") if the spread's own
                          AR(1) fit implies no mean reversion at all)
        "spread"       : pd.Series, the OLS residual series (for downstream Z-score use)
    """
    log_a = np.log(prices_a)
    log_b = np.log(prices_b)

    _t_stat, p_value, _crit_values = coint(log_a.values, log_b.values)

    hr = _hedge_ratio_and_spread(log_a, log_b)
    hl = _half_life(hr["spread"])

    return {"p_value": float(p_value), "hedge_ratio": hr["gamma"], "alpha": hr["alpha"],
            "half_life": hl, "spread": hr["spread"]}


def find_cointegrated_pairs(
    prices_df: pd.DataFrame,
    tickers: Optional[List[str]] = None,
    max_drift_pp: float = DEFAULT_MAX_DRIFT_PP,
    p_value_max: float = DEFAULT_P_VALUE_MAX,
    half_life_min: float = DEFAULT_HALF_LIFE_MIN,
    half_life_max: float = DEFAULT_HALF_LIFE_MAX,
    min_hedge_ratio: float = DEFAULT_MIN_HEDGE_RATIO,
    max_hedge_ratio: float = DEFAULT_MAX_HEDGE_RATIO,
) -> pd.DataFrame:
    """
    Scans every pair among `tickers` (default: all columns of `prices_df`)
    through the full gate sequence, in order (cheapest/least-informative
    checks first, so an expensive cointegration test is never run on a pair
    that was always going to be rejected on drift alone):

        1. 2Y drift-gate: |R_A(2Y) - R_B(2Y)| <= max_drift_pp
        2. Engle-Granger cointegration: p_value < p_value_max (via
           statsmodels.tsa.stattools.coint, NOT a plain adfuller() on
           residuals -- see module docstring)
        3. Half-life bound: half_life_min <= half_life <= half_life_max
        4. Hedge ratio sanity: min_hedge_ratio <= hedge_ratio <= max_hedge_ratio
           (rejects near-zero/negative/absurdly-large gamma -- a same-direction,
           comparably-scaled co-movement is what this relative-value
           mechanism actually needs; a negative or wildly-scaled gamma is a
           sign of a spurious regression, not a usable pair)

    Every gate is a HARD reject -- a pair failing any one of the four never
    reaches the returned table (unlike the buggy reference implementation
    this replaces, which computed p-value but never actually checked it).

    Parameters
    ----------
    prices_df : pd.DataFrame
        Daily close prices, one column per ticker, DatetimeIndex. Should
        already be the desired window (this function does not truncate --
        e.g. pass in a 5Y slice for a 5Y test, matching
        data.market_data.fetch_universe_prices's existing convention
        elsewhere in this project). NaN rows are dropped per-pair (pairwise
        intersection of valid dates), not globally, so one illiquid ticker
        doesn't shrink the window for every other pair -- same "Option A"
        philosophy already used by engine/risk.py's crash-overlap matrix.
    tickers : list of str or None
        Restrict the scan to this subset of `prices_df`'s columns. Defaults
        to every column.

    Returns
    -------
    pd.DataFrame, sorted by Score ascending (best first), columns:
        "Ticker A", "Ticker B", "Drift Diff 2Y [pp]", "P-Value", "Half-Life",
        "Hedge Ratio", "Score"
    Score = P-Value * Half-Life (confirmed design) -- lower is better:
    rewards both strong statistical evidence (low p-value) and a fast,
    practically tradeable reversion (low half-life).

    Empty DataFrame (same columns, zero rows) if no pair passes every gate --
    never raises just because nothing qualified.
    """
    if tickers is None:
        tickers = list(prices_df.columns)

    rows = []
    for tA, tB in itertools.combinations(tickers, 2):
        if tA not in prices_df.columns or tB not in prices_df.columns:
            continue
        pair_df = prices_df[[tA, tB]].dropna(how="any")
        if len(pair_df) < TRADING_DAYS_2Y + 30:  # potrzeba co najmniej okna 2Y + troche zapasu na regresje
            continue

        pa, pb = pair_df[tA], pair_df[tB]

        # --- Gate 1: drift 2Y ---
        ret_a_2y = float(pa.iloc[-1] / pa.iloc[-TRADING_DAYS_2Y] - 1.0)
        ret_b_2y = float(pb.iloc[-1] / pb.iloc[-TRADING_DAYS_2Y] - 1.0)
        drift_diff_pp = abs(ret_a_2y - ret_b_2y) * 100.0
        if drift_diff_pp > max_drift_pp:
            continue

        # --- Gate 2+3: kointegracja + half-life ---
        result = test_pair_cointegration(pa, pb)
        if result["p_value"] >= p_value_max:
            continue
        if not (half_life_min <= result["half_life"] <= half_life_max):
            continue

        # --- Gate 4: sensownosc hedge ratio ---
        gamma = result["hedge_ratio"]
        if not (min_hedge_ratio <= gamma <= max_hedge_ratio):
            continue

        rows.append({
            "Ticker A": tA, "Ticker B": tB, "Drift Diff 2Y [pp]": round(drift_diff_pp, 1),
            "P-Value": result["p_value"], "Half-Life": round(result["half_life"], 1),
            "Hedge Ratio": round(gamma, 3), "Score": result["p_value"] * result["half_life"],
        })

    columns = ["Ticker A", "Ticker B", "Drift Diff 2Y [pp]", "P-Value", "Half-Life", "Hedge Ratio", "Score"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("Score", ascending=True).reset_index(drop=True)
