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

TRADING_DAYS_2Y = 504  # 2 * 252, kept only as a general "at least ~2Y of data" minimum-length sanity floor

# Domyslne progi (confirmed design, 2026-09-08). Half-life zakres 5-63 sesji
# pochodzi wprost z raportu -- zostaje bez zmian na razie (miesieczny rytm
# rebalansu tego projektu moze docelowo uzasadniac zawezenie/rozszerzenie
# tego zakresu, ale to swiadoma, OSOBNA decyzja do podjecia PO potwierdzeniu,
# ze sam skaner liczy poprawnie -- nie zmieniane tutaj po cichu).
#
# UWAGA (2026-09-08, czwarty follow-up): 2Y drift-gate USUNIETY calkowicie --
# korelacja z Alpha wysza praktycznie zerowa (+0.02, patrz PROJECT_CONTEXT.md),
# co matematycznie potwierdzilo intuicje wlasciciela projektu: metryka byla
# PUNKTOWYM porownaniem (dzis vs dokladnie 2 lata temu) w jednostkach
# BEZWZGLEDNYCH (p.p.) na uniwersum spolek o skrajnie roznej skali 5-letniego
# wzrostu (400% vs 2500%) -- dwie spolki nie sa "rozjechane", bo jedna urosla
# duzo bardziej niz druga w wartosciach bezwzglednych, tylko jesli ich SCIEZKI
# CENOWE realnie sie od siebie oddalily w trakcie calego okna. Zastapione przez
# avg_relative_divergence_5y() -- patrz nizej -- jako zmienna WYLACZNIE
# informacyjna (do analizy korelacji), NIE bramka: liczba formalnych bramek
# spadla z 4 do 3 (kointegracja, half-life, hedge ratio).
DEFAULT_P_VALUE_MAX = 0.05
DEFAULT_HALF_LIFE_MIN = 5
DEFAULT_HALF_LIFE_MAX = 63
DEFAULT_MIN_HEDGE_RATIO = 0.3   # odrzuca hedge ratio bliskie 0 lub ujemne
DEFAULT_MAX_HEDGE_RATIO = 3.0   # odrzuca hedge ratio absurdalnie duze (skale cen niewspolmierne)


def avg_relative_divergence_5y(prices_a: pd.Series, prices_b: pd.Series) -> float:
    """
    Average relative divergence between two price PATHS over the full
    provided window (confirmed replacement for the removed 2Y drift gate,
    2026-09-08 fourth follow-up).

        norm_A(t) = P_A(t) / P_A(0),  norm_B(t) = P_B(t) / P_B(0)
        AvgRelativeDivergence = mean_t( |ln(norm_A(t)) - ln(norm_B(t))| )

    Both paths are rebased to 100 at the start of the window (exactly like
    the "100% A" / "100% B" traces already drawn on the strategy chart),
    then the log-difference between them is averaged across EVERY day in
    the window -- not a single point-in-time snapshot like the removed 2Y
    drift gate was. Using logs makes this symmetric (doesn't matter which
    ticker is "A" vs "B") and keeps it in the same "units" as the
    cointegration spread and hedge-ratio regression, which also operate on
    log prices.

    This directly fixes what made the removed metric fail (confirmed via
    its near-zero, +0.02, correlation with actual backtest Alpha): a
    single end-to-end percentage-point comparison conflates "these two
    paths diverged from each other" with "one of them happened to compound
    to a much bigger absolute number than the other" -- a universe with
    stocks ranging from +400% to +2500% over 5 years made ANY pair look
    "diverged" in raw percentage-point terms regardless of whether their
    price paths actually moved together or apart. Averaging the log-ratio
    across the whole window measures relative co-movement directly,
    independent of each stock's own absolute compounding.

    Deliberately informational only (not a gate) -- confirmed design: the
    project owner wants to see this variable's actual correlation with
    backtest performance before deciding whether a fixed threshold is
    warranted at all.
    """
    norm_a = prices_a / prices_a.iloc[0]
    norm_b = prices_b / prices_b.iloc[0]
    return float(np.abs(np.log(norm_a) - np.log(norm_b)).mean())


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


def evaluate_pair_diagnostics(
    prices_a: pd.Series, prices_b: pd.Series,
    p_value_max: float = DEFAULT_P_VALUE_MAX,
    half_life_min: float = DEFAULT_HALF_LIFE_MIN,
    half_life_max: float = DEFAULT_HALF_LIFE_MAX,
    min_hedge_ratio: float = DEFAULT_MIN_HEDGE_RATIO,
    max_hedge_ratio: float = DEFAULT_MAX_HEDGE_RATIO,
) -> Dict[str, object]:
    """
    Full gate-by-gate diagnostics for ONE pair, on an already-aligned
    (pairwise-dropna'd) price series pair. Does NOT short-circuit on the
    first failed gate -- confirmed requirement (2026-09-08 follow-up): the
    project owner wants visibility into NEAR-MISS pairs ("np. przeszły 2 z
    3 bramek"), which requires knowing the status of every gate, not just
    "rejected, reason unknown beyond the first failure".

    THREE formal gates now, not four (2026-09-08, fourth follow-up): the
    2Y drift gate was removed entirely after its correlation with actual
    backtest Alpha came back essentially zero (+0.02) -- a point-in-time,
    absolute-percentage-point comparison doesn't mean much across a
    universe where 5-year total returns range from +400% to +2500%.
    Replaced by `avg_relative_divergence_5y` (see that function's
    docstring), reported here PURELY as an informational field -- it does
    NOT participate in `gates_passed`/`all_passed` at all, per confirmed
    design ("chcę najpierw zobaczyć jaką będzie miał korelację").

    Returns a dict with every gate's pass/fail boolean AND its underlying
    value, plus "gates_passed" (0-3) and "all_passed" (bool, gates_passed==3):
        "avg_relative_divergence" (informational only, not a gate),
        "p_value", "coint_pass",
        "half_life", "half_life_pass",
        "hedge_ratio", "alpha", "hedge_ratio_pass",
        "spread" (pd.Series, for downstream chart/Z-score use),
        "gates_passed", "all_passed", "score" (p_value * half_life, only
        meaningful/comparable when all_passed -- see find_cointegrated_pairs)
    """
    avg_relative_divergence = avg_relative_divergence_5y(prices_a, prices_b)

    coint_result = test_pair_cointegration(prices_a, prices_b)
    p_value = coint_result["p_value"]
    half_life = coint_result["half_life"]
    hedge_ratio = coint_result["hedge_ratio"]

    coint_pass = p_value < p_value_max
    half_life_pass = half_life_min <= half_life <= half_life_max
    hedge_ratio_pass = min_hedge_ratio <= hedge_ratio <= max_hedge_ratio

    gates_passed = sum([coint_pass, half_life_pass, hedge_ratio_pass])

    return {
        "avg_relative_divergence": round(avg_relative_divergence, 4),
        "p_value": p_value, "coint_pass": coint_pass,
        "half_life": round(half_life, 1) if np.isfinite(half_life) else half_life, "half_life_pass": half_life_pass,
        "hedge_ratio": round(hedge_ratio, 3), "alpha": coint_result["alpha"], "hedge_ratio_pass": hedge_ratio_pass,
        "spread": coint_result["spread"],
        "gates_passed": gates_passed, "all_passed": gates_passed == 3,
        "score": (p_value * half_life) if np.isfinite(half_life) else float("inf"),
    }


def scan_universe_diagnostics(
    prices_df: pd.DataFrame,
    tickers: Optional[List[str]] = None,
    p_value_max: float = DEFAULT_P_VALUE_MAX,
    half_life_min: float = DEFAULT_HALF_LIFE_MIN,
    half_life_max: float = DEFAULT_HALF_LIFE_MAX,
    min_hedge_ratio: float = DEFAULT_MIN_HEDGE_RATIO,
    max_hedge_ratio: float = DEFAULT_MAX_HEDGE_RATIO,
) -> pd.DataFrame:
    """
    Full-universe scan returning EVERY pair with enough overlapping history
    (see TRADING_DAYS_2Y's minimum-length floor), each with its complete
    gate-by-gate diagnostics (see evaluate_pair_diagnostics) -- including
    pairs that did NOT pass all 3 gates, so a caller can inspect
    near-misses ("passed 2 of 3"), not just the fully-qualifying set.

    No cheap pre-filter anymore (2026-09-08, fourth follow-up): the former
    2Y-drift pre-filter was removed along with the gate itself (see
    avg_relative_divergence_5y's docstring for why) -- EVERY pair with
    sufficient history now pays for the full cointegration test. This is a
    real, accepted performance tradeoff (a large universe scan takes longer
    with no early-exit at all now) in exchange for not silently dropping
    pairs on a metric that turned out to have ~zero correlation with actual
    backtest performance.

    Returns
    -------
    pd.DataFrame, sorted by gates_passed descending then Score ascending
    (best/closest-to-qualifying first), with columns:
        "Ticker A", "Ticker B", "Gates Passed", "All Passed",
        "Avg Relative Divergence" (informational, NOT a gate),
        "P-Value", "Coint Pass", "Half-Life", "Half-Life Pass",
        "Hedge Ratio", "Hedge Ratio Pass", "Score"
    Empty DataFrame (same columns) if no pair has enough shared history --
    never raises.
    """
    if tickers is None:
        tickers = list(prices_df.columns)

    columns = ["Ticker A", "Ticker B", "Gates Passed", "All Passed", "Avg Relative Divergence",
               "P-Value", "Coint Pass", "Half-Life", "Half-Life Pass", "Hedge Ratio", "Hedge Ratio Pass", "Score"]
    rows = []
    for tA, tB in itertools.combinations(tickers, 2):
        if tA not in prices_df.columns or tB not in prices_df.columns:
            continue
        pair_df = prices_df[[tA, tB]].dropna(how="any")
        if len(pair_df) < TRADING_DAYS_2Y + 30:
            continue
        pa, pb = pair_df[tA], pair_df[tB]

        diag = evaluate_pair_diagnostics(pa, pb, p_value_max, half_life_min, half_life_max, min_hedge_ratio, max_hedge_ratio)
        rows.append({
            "Ticker A": tA, "Ticker B": tB, "Gates Passed": diag["gates_passed"], "All Passed": diag["all_passed"],
            "Avg Relative Divergence": diag["avg_relative_divergence"],
            "P-Value": diag["p_value"], "Coint Pass": diag["coint_pass"],
            "Half-Life": diag["half_life"], "Half-Life Pass": diag["half_life_pass"],
            "Hedge Ratio": diag["hedge_ratio"], "Hedge Ratio Pass": diag["hedge_ratio_pass"],
            "Score": diag["score"],
        })

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["Gates Passed", "Score"], ascending=[False, True]).reset_index(drop=True)


def find_cointegrated_pairs(
    prices_df: pd.DataFrame,
    tickers: Optional[List[str]] = None,
    p_value_max: float = DEFAULT_P_VALUE_MAX,
    half_life_min: float = DEFAULT_HALF_LIFE_MIN,
    half_life_max: float = DEFAULT_HALF_LIFE_MAX,
    min_hedge_ratio: float = DEFAULT_MIN_HEDGE_RATIO,
    max_hedge_ratio: float = DEFAULT_MAX_HEDGE_RATIO,
) -> pd.DataFrame:
    """
    The original, fully-qualifying-only screener -- now a thin filter over
    scan_universe_diagnostics's richer output (single source of truth for
    the gate logic).

    Every gate is a HARD reject -- a pair failing any one of the THREE
    (cointegration p-value, half-life, hedge-ratio sanity -- the former 2Y
    drift gate was removed entirely, 2026-09-08 fourth follow-up, see
    avg_relative_divergence_5y) never reaches the returned table.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Daily close prices, one column per ticker, DatetimeIndex.
    tickers : list of str or None
        Restrict the scan to this subset of `prices_df`'s columns.

    Returns
    -------
    pd.DataFrame, sorted by Score ascending (best first), columns:
        "Ticker A", "Ticker B", "P-Value", "Half-Life", "Hedge Ratio", "Score"
    Empty DataFrame (same columns) if no pair passes every gate.
    """
    full = scan_universe_diagnostics(prices_df, tickers, p_value_max, half_life_min, half_life_max, min_hedge_ratio, max_hedge_ratio)
    columns = ["Ticker A", "Ticker B", "P-Value", "Half-Life", "Hedge Ratio", "Score"]
    if full.empty:
        return pd.DataFrame(columns=columns)
    qualifying = full[full["All Passed"]][columns].sort_values("Score", ascending=True).reset_index(drop=True)
    return qualifying


DEFAULT_ZSCORE_LOOKBACK = 63     # ~1 kwartal, rolling okno do zywego Z-score (odrebne od statycznej regresji uzywanej w screeningu)
DEFAULT_ENTRY_Z = 1.5
DEFAULT_EXIT_Z = 0.25
DEFAULT_FAVOUR_WEIGHT = 0.80
DEFAULT_FEE_BPS = 10.0           # 10 bps prowizji i poslizgu, zastosowane PRZY KAZDEJ zmianie stanu


def simulate_pair_strategy(
    prices_a: pd.Series, prices_b: pd.Series, hedge_ratio: float,
    entry_z: float = DEFAULT_ENTRY_Z, exit_z: float = DEFAULT_EXIT_Z,
    favour_weight: float = DEFAULT_FAVOUR_WEIGHT, zscore_lookback: int = DEFAULT_ZSCORE_LOOKBACK,
    fee_bps: float = DEFAULT_FEE_BPS, initial_cash: float = 100.0,
) -> Dict[str, object]:
    """
    Discrete-event, threshold-switching Long-Only backtest for ONE pair --
    an interactive DIAGNOSTIC/exploration tool ("sprawdzić jak to się
    zachowuje" przy różnych progach, confirmed 2026-09-08), distinct from
    and NOT a preview of the eventual monthly theta*tanh(-Z) mu-adjustment
    mechanism (that stays a separate, softer, rebalance-cadence-aligned
    step -- this function is a daily-granularity illustration of the raw
    Z-score's behavior, useful for understanding a pair before deciding how
    much weight to give its signal, not the production execution path).

    State machine (three states, matching the confirmed 80/20-style design):
        Z <= -entry_z   -> FAVOUR_A  (weight_A = favour_weight)
        Z >=  entry_z   -> FAVOUR_B  (weight_A = 1 - favour_weight)
        |Z| <= exit_z   -> NEUTRAL   (weight_A = 0.5)
        (between exit_z and entry_z, in either direction: state doesn't
        change -- avoids flapping right at the threshold edges)

    Z-score uses a ROLLING window (zscore_lookback trading days) on the
    OLS spread log(P_A) - hedge_ratio*log(P_B) -- deliberately different
    from the STATIC, full-sample regression used by
    test_pair_cointegration/evaluate_pair_diagnostics for the screening
    p-value (that one needs a fixed window to be a valid statistical test;
    this one is a live/adaptive trading signal, where adapting to recent
    spread dynamics is the point). `hedge_ratio` is still the FIXED value
    from the screener, though -- only the mean/std normalizing it rolls.

    Execution: physical share counts (NOT daily-rebalanced weight
    fractions) -- this is the fix for a real bug ("Shannon's Demon")
    identified in the reference implementation this project audited: naive
    daily `w_A*R_A + w_B*R_B` cumulative-product accounting implicitly
    forces a FREE daily rebalance, which manufactures return purely from
    volatility (the passive 50/50 benchmark can end up beating BOTH
    components held individually, which is not a real phenomenon). Here,
    shares are only ever bought/sold on a STATE CHANGE, with `fee_bps`
    applied to the traded value at that moment -- between state changes,
    the position drifts exactly like a real physical holding.

    Parameters
    ----------
    prices_a, prices_b : pd.Series
        Already-aligned (pairwise-dropna'd) daily close prices, same index.
    hedge_ratio : float
        Fixed hedge ratio (gamma) from the screener -- see
        evaluate_pair_diagnostics / test_pair_cointegration.
    entry_z, exit_z : float
        Z-score thresholds for entering a favoured state / returning to neutral.
    favour_weight : float
        Weight given to the favoured ticker when |Z| crosses entry_z (e.g.
        0.80 for an 80/20 split; the other ticker gets 1 - favour_weight).
    zscore_lookback : int
        Rolling window (trading days) for the live Z-score's mean/std.
    fee_bps : float
        Transaction cost (basis points) applied to traded value on every
        state change.
    initial_cash : float
        Starting capital for every curve (all start at the same base).

    Returns
    -------
    dict with keys:
        "dates"              : list of ISO date strings
        "equity_strategy"    : list[float] -- the dynamic threshold-switching curve
        "equity_benchmark"   : list[float] -- passive 50/50 fixed-share Buy & Hold
        "equity_100a"        : list[float] -- 100% A Buy & Hold
        "equity_100b"        : list[float] -- 100% B Buy & Hold
        "zscore"              : list[float] -- the rolling Z-score series
        "weight_a"            : list[float] -- strategy's realized weight in A over time
        "n_transitions"       : int -- how many state changes occurred (for context:
                                 an unreasonably high count at tight thresholds signals
                                 excessive turnover/fee drag, worth seeing directly)
    """
    log_spread = np.log(prices_a) - hedge_ratio * np.log(prices_b)
    roll_mean = log_spread.rolling(zscore_lookback).mean()
    roll_std = log_spread.rolling(zscore_lookback).std()
    zscore = (log_spread - roll_mean) / roll_std

    df = pd.DataFrame({"pA": prices_a, "pB": prices_b, "Z": zscore}).dropna()
    if df.empty:
        return {"dates": [], "equity_strategy": [], "equity_benchmark": [], "equity_100a": [], "equity_100b": [],
                "zscore": [], "weight_a": [], "n_transitions": 0}

    norm_a = (df["pA"] / df["pA"].iloc[0]) * initial_cash
    norm_b = (df["pB"] / df["pB"].iloc[0]) * initial_cash
    equity_benchmark = 0.5 * norm_a + 0.5 * norm_b  # NIE cumprod() -- patrz docstring

    shares_a = (initial_cash * 0.5) / df["pA"].iloc[0]
    shares_b = (initial_cash * 0.5) / df["pB"].iloc[0]
    fee_frac = fee_bps / 10000.0
    current_state = "NEUTRAL"
    n_transitions = 0

    equity_strategy, weight_a_hist = [], []
    for t_idx, (pa, pb, z) in enumerate(zip(df["pA"].values, df["pB"].values, df["Z"].values)):
        new_state = current_state
        if z <= -entry_z:
            new_state = "FAVOUR_A"
        elif z >= entry_z:
            new_state = "FAVOUR_B"
        elif abs(z) <= exit_z:
            new_state = "NEUTRAL"

        port_val = shares_a * pa + shares_b * pb
        if new_state != current_state and t_idx > 0:
            target_wa = favour_weight if new_state == "FAVOUR_A" else ((1.0 - favour_weight) if new_state == "FAVOUR_B" else 0.5)
            port_val *= (1.0 - fee_frac)
            shares_a = (port_val * target_wa) / pa
            shares_b = (port_val * (1.0 - target_wa)) / pb
            current_state = new_state
            n_transitions += 1

        cur_val = shares_a * pa + shares_b * pb
        equity_strategy.append(cur_val)
        weight_a_hist.append((shares_a * pa) / cur_val if cur_val > 0 else 0.5)

    return {
        "dates": [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in df.index],
        "equity_strategy": equity_strategy, "equity_benchmark": list(equity_benchmark.values),
        "equity_100a": list(norm_a.values), "equity_100b": list(norm_b.values),
        "zscore": list(df["Z"].values), "weight_a": weight_a_hist, "n_transitions": n_transitions,
    }


def _max_drawdown_pct(equity_curve: List[float]) -> float:
    """Worst peak-to-trough decline (%) in an equity curve. 0.0 for an
    empty/constant/monotonically-increasing curve."""
    if not equity_curve:
        return 0.0
    arr = np.array(equity_curve)
    running_max = np.maximum.accumulate(arr)
    drawdowns = (arr - running_max) / running_max
    return float(drawdowns.min() * 100.0)


def run_backtest_batch(
    prices_df: pd.DataFrame, pairs_df: pd.DataFrame,
    entry_z: float = DEFAULT_ENTRY_Z, exit_z: float = DEFAULT_EXIT_Z,
    favour_weight: float = DEFAULT_FAVOUR_WEIGHT, zscore_lookback: int = DEFAULT_ZSCORE_LOOKBACK,
    fee_bps: float = DEFAULT_FEE_BPS,
) -> pd.DataFrame:
    """
    Runs simulate_pair_strategy for EVERY pair in `pairs_df` (expected to
    have "Ticker A", "Ticker B", "Hedge Ratio" columns -- exactly what
    scan_universe_diagnostics returns, so its full near-miss output can be
    fed straight in), under the SAME threshold settings for every pair, and
    ranks the result by ACTUAL backtest performance rather than by gate-pass
    count. Confirmed motivation (2026-09-08, second follow-up): the
    screening gates test whether a stable statistical equilibrium exists,
    which is necessary but not sufficient for a profitable Long-Only
    threshold-switching strategy -- a pair can be "significantly
    cointegrated" with tiny, rarely-triggered spread swings that barely
    beat the passive benchmark, while a pair that only weakly qualifies (or
    doesn't formally qualify at all) can still show strong backtest
    performance if its Z-score swings are large and its components share a
    correlated growth trend the Long-Only mechanism can ride.

    All of `pairs_df`'s original columns are carried through into the
    output UNCHANGED, plus new columns:
        "Final Equity", "Relative Alpha [%]" -- CONFIRMED redefinition
        (2026-09-08, fourth follow-up): = (final_strategy / best_alternative
        - 1) * 100, where best_alternative = max(final_benchmark,
        final_100a, final_100b) -- the BEST of the three non-dynamic
        curves, not always the 50/50 benchmark specifically. This directly
        answers "did the dynamic allocation beat even the best single
        stock in hindsight", a harder bar than beating the passive basket.
        Already genuinely relative by construction (a ratio, not a raw
        equity-unit difference) -- verified against confirmed worked
        examples (2000 vs 1500 -> +33.3%; 500 vs 335 -> +49.3%). The old
        "[pp]" label was misleading (it read as raw percentage points of
        difference despite always having been ratio-based) -- renamed to
        "[%]" for clarity, no formula change beyond the new denominator.
        "Max Drawdown [%]", "N Transitions", "Z-Score Std", "Raw Spread
        Volatility" (annualized std of the daily log-ratio log(P_A/P_B),
        UNNORMALIZED -- kept alongside "Z-Score Std" since Z-scoring is
        scale-invariant by construction and can look similar across pairs
        with very different raw amplitude, confirmed via a synthetic
        5-pair test), "Days In Lead [%]" (fraction of days the dynamic
        strategy's equity was >= the best of the other three curves, ON
        THAT DAY -- not just at the end), and "Composite Score" (see below).

    "Composite Score" -- confirmed ranking formula (2026-09-08, fourth
    follow-up): 50% each of "Relative Alpha [%]" and "Days In Lead [%]",
    combined via PERCENTILE RANK (`pandas.Series.rank(pct=True)`) rather
    than a raw weighted sum of the two values directly. This is a
    deliberate choice, not an arbitrary one: the two metrics live on very
    different scales (Alpha can range from roughly -90% to well over
    +1000%; Days In Lead is bounded to [0, 100]) -- a raw
    `0.5*Alpha + 0.5*DaysInLead` would be completely dominated by whichever
    metric happens to have the larger numeric range for a given batch,
    regardless of which one is actually more informative. Rank-based
    combination is scale-invariant: each pair's contribution from each
    metric is its RELATIVE STANDING within this batch (0 to 1), so both
    halves carry genuinely equal weight regardless of the units involved.

    Sorted by "Composite Score" descending (best first) -- NOT by
    Score/gates-passed, and not by Alpha alone.

    Pairs where the underlying price data can't support a full backtest
    (e.g. too few overlapping observations for the rolling Z-score window)
    are silently skipped, not included with null placeholders.
    """
    rows = []
    for _, row in pairs_df.iterrows():
        tA, tB, hedge_ratio = row["Ticker A"], row["Ticker B"], row["Hedge Ratio"]
        if tA not in prices_df.columns or tB not in prices_df.columns:
            continue
        pair_df = prices_df[[tA, tB]].dropna(how="any")
        pa, pb = pair_df[tA], pair_df[tB]

        result = simulate_pair_strategy(pa, pb, hedge_ratio=hedge_ratio, entry_z=entry_z, exit_z=exit_z,
                                         favour_weight=favour_weight, zscore_lookback=zscore_lookback, fee_bps=fee_bps)
        if not result["dates"]:
            continue

        final_strategy = result["equity_strategy"][-1]
        final_benchmark = result["equity_benchmark"][-1]
        final_100a = result["equity_100a"][-1]
        final_100b = result["equity_100b"][-1]
        best_alternative = max(final_benchmark, final_100a, final_100b)
        relative_alpha_pct = (final_strategy / best_alternative - 1.0) * 100.0 if best_alternative > 0 else 0.0

        strat_arr = np.array(result["equity_strategy"])
        others_max = np.maximum.reduce([np.array(result["equity_benchmark"]), np.array(result["equity_100a"]), np.array(result["equity_100b"])])
        days_in_lead_pct = float(np.mean(strat_arr >= others_max) * 100.0)

        out_row = dict(row)
        out_row.update({
            "Final Equity": round(final_strategy, 1),
            "Relative Alpha [%]": round(relative_alpha_pct, 2),
            "Max Drawdown [%]": round(_max_drawdown_pct(result["equity_strategy"]), 1),
            "N Transitions": result["n_transitions"],
            "Z-Score Std": round(float(np.std(result["zscore"])), 3) if result["zscore"] else 0.0,
            "Raw Spread Volatility": round(float(np.log(pair_df[tA] / pair_df[tB]).diff().std() * np.sqrt(252)), 4),
            "Days In Lead [%]": round(days_in_lead_pct, 1),
        })
        rows.append(out_row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    alpha_rank = out["Relative Alpha [%]"].rank(pct=True)
    lead_rank = out["Days In Lead [%]"].rank(pct=True)
    out["Composite Score"] = round(0.5 * alpha_rank + 0.5 * lead_rank, 4)
    return out.sort_values("Composite Score", ascending=False).reset_index(drop=True)


def compute_correlations(df: pd.DataFrame, target_col: str, candidate_cols: List[str]) -> pd.Series:
    """
    Pearson correlation of each column in `candidate_cols` against
    `target_col` -- the direct, numeric answer to "which of these variables
    actually explains the outperformance" (confirmed motivation,
    2026-09-08 second follow-up), rather than eyeballing a table.

    Generic over whatever columns are present -- does not know or care
    whether a candidate column came from the screener's own gate
    diagnostics (P-Value, Half-Life, ...), from run_backtest_batch itself
    (Z-Score Std, N Transitions, ...), or was attached by the CALLER from
    an entirely different source (e.g. a "Same Sector" boolean the UI layer
    adds from data.universe_store -- deliberately not computed here, since
    engine/ never imports data/, per the one-directional dependency rule).

    Returns a pd.Series indexed by candidate column name, sorted by
    absolute correlation strength descending (most explanatory first,
    regardless of sign). Columns missing from `df`, or with zero variance
    (correlation undefined), are silently excluded -- not returned as NaN.
    """
    correlations = {}
    for col in candidate_cols:
        if col not in df.columns or col == target_col:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.nunique(dropna=True) < 2:
            continue
        corr = series.corr(pd.to_numeric(df[target_col], errors="coerce"))
        if pd.notna(corr):
            correlations[col] = float(corr)
    return pd.Series(correlations).sort_values(key=lambda s: s.abs(), ascending=False)