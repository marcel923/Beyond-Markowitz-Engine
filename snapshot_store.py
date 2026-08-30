"""
snapshot_store.py
==================
Persistence layer for saved portfolio snapshots ("Forward-Testing / Out-of-Sample
Logging System"). Pure Python + a bit of yfinance for live price lookups -- no Dash
code lives here on purpose, same modular split as tps_solver.py.

Storage format
--------------
A single JSON file (`DEFAULT_FILE_PATH`, default "saved_portfolios.json") holding a
list of snapshot records, each shaped like:

    {
        "snapshot_id":        "2026-07-31_14-30",
        "snapshot_name":      "Defensive_Lambda0.8_v1",
        "created_at":         "2026-07-31T14:30:05",
        "parameters":         {"lambda": 0.8, "gamma": 1.5, ...},
        "fundamental_inputs": [{"Ticker": "AAPL", ...}, ...],   # full Stage 3 payload
        "final_weights":      {"AAPL": 0.18, "MSFT": 0.12, ...},
        "cluster_of":         {"AAPL": 1, "MSFT": 2, ...},
        "entry_prices":       {"AAPL": 227.5, "MSFT": 415.2, ..., "SPY": 560.1}
    }

Writes are done via write-to-temp-then-os.replace to avoid leaving a truncated/corrupt
file behind if the process is killed mid-write.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

DEFAULT_FILE_PATH = "saved_portfolios.json"
BENCHMARK_TICKER = "SPY"


# ---------------------------------------------------------------------------
# Low-level file I/O
# ---------------------------------------------------------------------------

def _read_all(file_path: str = DEFAULT_FILE_PATH) -> List[dict]:
    """Returns [] if the file doesn't exist yet or is malformed -- never raises."""
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _write_all(snapshots: List[dict], file_path: str = DEFAULT_FILE_PATH) -> None:
    tmp_path = file_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(snapshots, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, file_path)


def make_snapshot_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M")


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def save_snapshot(
    snapshot_name: str,
    parameters: dict,
    fundamental_inputs: list,
    final_weights: dict,
    cluster_of: dict,
    entry_prices: dict,
    z_scores: Optional[dict] = None,
    file_path: str = DEFAULT_FILE_PATH,
) -> dict:
    """
    Appends a new snapshot record and persists it to disk. Returns the created record.

    `z_scores` (optional): per-ticker Z_TR,i tail-risk Z-scores frozen at save time
    (Section 4.1). These are lambda-INDEPENDENT, so storing them lets a later "what-if"
    sandbox recompute P_i = exp(new_lambda * max(0, Z)) correctly for any lambda the user
    tries -- without needing to re-download 5 years of price history. Old snapshots saved
    before this field existed simply won't have it; callers should treat a missing/absent
    z_scores dict as "no frozen risk data available" and degrade gracefully (e.g. assume
    Z=0 / P_i=1 baseline), not crash.
    """
    snapshots = _read_all(file_path)

    base_id = make_snapshot_id()
    existing_ids = {s.get("snapshot_id") for s in snapshots}
    snapshot_id = base_id
    suffix = 1
    while snapshot_id in existing_ids:
        suffix += 1
        snapshot_id = f"{base_id}_{suffix}"

    record = {
        "snapshot_id": snapshot_id,
        "snapshot_name": snapshot_name.strip() if snapshot_name and snapshot_name.strip() else f"Portfolio_{snapshot_id}",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "parameters": parameters,
        "fundamental_inputs": fundamental_inputs,
        "final_weights": final_weights,
        "cluster_of": cluster_of,
        "entry_prices": entry_prices,
        "z_scores": z_scores or {},
    }
    snapshots.append(record)
    _write_all(snapshots, file_path)
    return record


def list_snapshots(file_path: str = DEFAULT_FILE_PATH) -> List[dict]:
    """Newest first -- most useful ordering for a dropdown."""
    snapshots = _read_all(file_path)
    return sorted(snapshots, key=lambda s: s.get("created_at", ""), reverse=True)


def get_snapshot(snapshot_id: str, file_path: str = DEFAULT_FILE_PATH) -> Optional[dict]:
    for s in _read_all(file_path):
        if s.get("snapshot_id") == snapshot_id:
            return s
    return None


def delete_snapshot(snapshot_id: str, file_path: str = DEFAULT_FILE_PATH) -> bool:
    snapshots = _read_all(file_path)
    new_list = [s for s in snapshots if s.get("snapshot_id") != snapshot_id]
    if len(new_list) == len(snapshots):
        return False
    _write_all(new_list, file_path)
    return True


def rename_snapshot(snapshot_id: str, new_name: str, file_path: str = DEFAULT_FILE_PATH) -> bool:
    if not new_name or not new_name.strip():
        return False
    snapshots = _read_all(file_path)
    found = False
    for s in snapshots:
        if s.get("snapshot_id") == snapshot_id:
            s["snapshot_name"] = new_name.strip()
            found = True
            break
    if found:
        _write_all(snapshots, file_path)
    return found


# ---------------------------------------------------------------------------
# Live price lookups + forward-return math
# ---------------------------------------------------------------------------

def fetch_current_prices(tickers: List[str]) -> Dict[str, float]:
    """
    Single batched yfinance call for a list of tickers -> {ticker: last_close_price}.
    Tickers that fail to resolve are simply absent from the returned dict (defensive --
    callers should handle missing tickers rather than assume full coverage).

    Uses period="5d" (not "1d") so a weekend/holiday call still finds a recent close,
    and explicitly checks for the MultiIndex-columns quirk yfinance has with
    group_by="ticker" (same fix applied elsewhere in this project for the single-ticker case).
    """
    import yfinance as yf

    tickers = list(dict.fromkeys(tickers))  # dedupe, preserve order
    if not tickers:
        return {}

    try:
        df = yf.download(tickers, period="5d", group_by="ticker", threads=False, auto_adjust=True, progress=False)
    except Exception:
        return {}

    if df is None or df.empty:
        return {}

    prices: Dict[str, float] = {}
    is_multi = isinstance(df.columns, pd.MultiIndex)
    for t in tickers:
        try:
            if is_multi:
                if t not in df.columns.get_level_values(0):
                    continue
                s = df[t]["Close"].dropna()
            else:
                s = df["Close"].dropna()
            if len(s):
                prices[t] = float(s.iloc[-1])
        except Exception:
            continue
    return prices


def fetch_price_history(tickers: List[str], start_date: str) -> "pd.DataFrame":
    """
    Batched yfinance call fetching DAILY close prices for a list of tickers from
    `start_date` (YYYY-MM-DD) through today. Used to reconstruct real forward equity
    curves since a snapshot's creation date.

    Returns a DataFrame indexed by date, one column per ticker that resolved successfully
    (missing/failed tickers are simply absent as columns -- callers should check which of
    their requested tickers actually made it into the result). Returns an empty DataFrame
    (not an exception) on total failure, so callers can handle "no data" uniformly.
    """
    import yfinance as yf

    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return pd.DataFrame()

    try:
        df = yf.download(tickers, start=start_date, group_by="ticker", threads=False, auto_adjust=True, progress=False)
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    is_multi = isinstance(df.columns, pd.MultiIndex)
    prices = pd.DataFrame(index=df.index)
    for t in tickers:
        try:
            if is_multi:
                if t not in df.columns.get_level_values(0):
                    continue
                prices[t] = df[t]["Close"]
            else:
                if "Close" not in df.columns:
                    continue
                prices[t] = df["Close"]
        except Exception:
            continue

    return prices.dropna(how="all")



def compute_forward_return(weights: Dict[str, float], entry_prices: Dict[str, float],
                            current_prices: Dict[str, float]) -> Tuple[Optional[float], float]:
    """
    Return_P = sum_i w_i * (P_current,i - P0,i) / P0,i

    Tickers missing from either price dict (e.g. a live fetch failure for one name) are
    skipped rather than crashing the whole calculation. Returns (return_value, weight_coverage)
    where weight_coverage is the fraction of total portfolio weight actually included --
    callers should warn the user if this is meaningfully below 1.0 (stale/partial read).

    Returns (None, 0.0) if no ticker could be matched at all.
    """
    total = 0.0
    weight_used = 0.0
    for t, w in weights.items():
        if t not in entry_prices or t not in current_prices:
            continue
        p0 = entry_prices[t]
        pc = current_prices[t]
        if not p0:
            continue
        total += w * (pc - p0) / p0
        weight_used += w

    if weight_used <= 1e-9:
        return None, 0.0
    return total, weight_used


def evaluate_snapshot(snapshot: dict) -> dict:
    """
    Full forward-tracking evaluation for one saved snapshot: fetches live prices for the
    snapshot's tickers PLUS the SPY benchmark in a single batched call, then computes both
    the portfolio's unrealized forward return and the SPY benchmark return over the same
    holding period.

    Returns a dict with keys:
        "portfolio_return"   : float or None
        "weight_coverage"     : float, fraction of portfolio weight with a valid live price
        "spy_return"          : float or None
        "current_prices"      : dict, live prices actually fetched
        "missing_tickers"     : list, tickers that could not be priced live
        "holding_days"        : int, days since snapshot creation
        "error"               : str or None
    """
    weights = snapshot.get("final_weights", {})
    entry_prices = snapshot.get("entry_prices", {})
    tickers = list(weights.keys())
    fetch_list = tickers + [BENCHMARK_TICKER]

    current_prices = fetch_current_prices(fetch_list)
    if not current_prices:
        return {
            "portfolio_return": None, "weight_coverage": 0.0, "spy_return": None,
            "current_prices": {}, "missing_tickers": fetch_list,
            "holding_days": _holding_days(snapshot), "error": "yfinance nie zwrócił żadnych cen (rate limit / brak połączenia?)."
        }

    portfolio_return, weight_coverage = compute_forward_return(weights, entry_prices, current_prices)

    spy_return = None
    if BENCHMARK_TICKER in current_prices and BENCHMARK_TICKER in entry_prices and entry_prices[BENCHMARK_TICKER]:
        spy_p0 = entry_prices[BENCHMARK_TICKER]
        spy_pc = current_prices[BENCHMARK_TICKER]
        spy_return = (spy_pc - spy_p0) / spy_p0

    missing = [t for t in fetch_list if t not in current_prices]

    return {
        "portfolio_return": portfolio_return, "weight_coverage": weight_coverage, "spy_return": spy_return,
        "current_prices": current_prices, "missing_tickers": missing,
        "holding_days": _holding_days(snapshot), "error": None
    }


def holding_days_since(snapshot: dict) -> int:
    """Public wrapper -- days elapsed since a snapshot's created_at."""
    return _holding_days(snapshot)

def _holding_days(snapshot: dict) -> int:
    created_at = snapshot.get("created_at")
    if not created_at:
        return 0
    try:
        created = datetime.fromisoformat(created_at)
        return max((datetime.now() - created).days, 0)
    except (ValueError, TypeError):
        return 0