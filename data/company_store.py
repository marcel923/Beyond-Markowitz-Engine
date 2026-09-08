"""
data/company_store.py
=======================
Persistence layer for per-company historical fundamental data (Etap 1,
PROJECT_CONTEXT.md v2 storage spec; extended Etap 4 follow-up, 2026-09-07,
with per-horizon projection-model tuning profiles). One JSON file per
ticker under storage/company_history/, holding an append-only log of
fundamental snapshots over time -- this is the raw material both for the
Company Dossier module (target-price-vs-actual-price chart, single-asset
forward projection cone) and, much later, for training an ML expected-return
model (Etap 6 roadmap): the whole reason this gets its own store now, well
before any ML work starts, is that history only accumulates from whenever
recording starts. There is no way to backfill it later -- EXCEPT for
entries the user manually backfills with a past date via the Research
module's DatePickerSingle (Etap 4 follow-up), which is exactly why
`append_entry` takes an explicit `entry_date` rather than always stamping
"now": a user who already knows a company's fundamentals from 2025-09-01
can enter that historical point directly, instead of only ever being able
to start today.

Storage format
--------------
storage/company_history/{TICKER}.json holds a single JSON OBJECT (not a
bare list, since the Etap 4 follow-up):

    {
        "entries": [
            {
                "Date":             "2026-09-01",
                "P0":               217.44,
                "Target_Consensus": 300.0,
                "Target_High":      400.0,
                "Target_Low":       100.0,
                "N_analysts":       12,
                "EPS_CAGR":         12.0,
                "EPS_Rev_90d":      0.0
            },
            ...
        ],
        "horizon_profiles": {
            "1M": {"alpha": 0.15, "gamma": 0.20, "kappa": 1.60, "eta": 1.00},
            "3M": {...}, "6M": {...}, "1Y": {...}
        }
    }

`entries` is always kept sorted chronologically by "Date" (oldest first) --
required so a plotted price/target history connects points in time order
regardless of the order they were entered/backfilled in. One entry per
Date: re-recording the same Date UPDATES that entry in place (upsert)
rather than appending a duplicate.

`horizon_profiles` holds a saved {alpha, gamma, kappa, eta} tuning override
per horizon (see engine/single_asset.py) for THIS specific company -- e.g.
"NVDA's 1Y eta should be 1.15 because it's historically a beat-and-raise
name". A horizon with no saved profile is simply absent from this dict;
`get_horizon_profile` returns None for it, and the engine-level default
(engine.single_asset.DEFAULT_HORIZON_PARAMS) is what the UI falls back to
-- this module deliberately does NOT import engine/single_asset.py to
supply that default itself, keeping the one-directional dependency rule
(data/ never imports engine/) intact; the UI layer is what bridges the two.

Backward compatibility: a file written before this Etap 4 follow-up is a
bare JSON list (the old format, no wrapping object). Reading such a file
transparently treats it as {"entries": <that list>, "horizon_profiles": {}}
-- no migration script or one-time conversion needed; the next write to
that file naturally upgrades it to the new wrapped format.

Same atomic write-to-temp-then-os.replace pattern as the other data/ stores.
No Dash / UI code lives here on purpose.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime
from typing import Dict, List, Optional

DEFAULT_HISTORY_DIR = "storage/company_history"
VALID_HORIZONS = ("1M", "3M", "6M", "1Y")


# ---------------------------------------------------------------------------
# Low-level file I/O
# ---------------------------------------------------------------------------

def _ticker_file_path(ticker: str, base_dir: str = DEFAULT_HISTORY_DIR) -> str:
    """Sanitizes the ticker for use as a filename. Most tickers (NVDA,
    005930.KS, LYC.AX) are already filesystem-safe as-is; this only guards
    against the rare ticker convention that includes a path separator."""
    safe_ticker = re.sub(r'[\\/]', '_', ticker.strip().upper())
    return os.path.join(base_dir, f"{safe_ticker}.json")


def _read_raw(ticker: str, base_dir: str = DEFAULT_HISTORY_DIR) -> Dict[str, object]:
    """
    Returns {"entries": [...], "horizon_profiles": {...}}, regardless of
    whether the file on disk is the new wrapped-object format or an old
    bare-list file (backward compat, see module docstring). Never raises --
    missing/corrupt files degrade to an empty structure.
    """
    path = _ticker_file_path(ticker, base_dir)
    if not os.path.exists(path):
        return {"entries": [], "horizon_profiles": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"entries": [], "horizon_profiles": {}}

    if isinstance(data, list):  # old bare-list format
        return {"entries": data, "horizon_profiles": {}}
    if isinstance(data, dict):
        return {"entries": data.get("entries", []) or [], "horizon_profiles": data.get("horizon_profiles", {}) or {}}
    return {"entries": [], "horizon_profiles": {}}


def _write_raw(ticker: str, raw: Dict[str, object], base_dir: str = DEFAULT_HISTORY_DIR) -> None:
    os.makedirs(base_dir, exist_ok=True)
    path = _ticker_file_path(ticker, base_dir)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# CRUD -- fundamental entries
# ---------------------------------------------------------------------------

def append_entry(
    ticker: str,
    p0: float,
    target_consensus: float,
    target_high: float,
    target_low: float,
    n_analysts: int,
    eps_cagr: float,
    eps_rev_90d: float,
    entry_date: Optional[str] = None,
    base_dir: str = DEFAULT_HISTORY_DIR,
) -> dict:
    """
    Records a fundamental data point for `ticker` under `entry_date`
    (ISO "YYYY-MM-DD"; defaults to today if not given). `entry_date` need
    NOT be today -- passing a past date backfills historical data (Etap 4
    follow-up). If an entry for that exact date already exists, it is
    REPLACED (upsert), not duplicated. `entries` is always re-sorted
    chronologically by Date after every write, so a backfilled past date
    lands in the correct position for charting regardless of entry order.

    Returns the resulting entry.
    """
    entry_date = entry_date or date.today().isoformat()
    entry = {
        "Date": entry_date, "P0": float(p0), "Target_Consensus": float(target_consensus),
        "Target_High": float(target_high), "Target_Low": float(target_low),
        "N_analysts": int(n_analysts), "EPS_CAGR": float(eps_cagr), "EPS_Rev_90d": float(eps_rev_90d),
    }

    raw = _read_raw(ticker, base_dir)
    entries = raw["entries"]
    for i, e in enumerate(entries):
        if e.get("Date") == entry_date:
            entries[i] = entry
            break
    else:
        entries.append(entry)

    entries.sort(key=lambda e: e.get("Date", ""))
    raw["entries"] = entries
    _write_raw(ticker, raw, base_dir)
    return entry


def get_history(ticker: str, base_dir: str = DEFAULT_HISTORY_DIR) -> List[dict]:
    """Full history for one ticker, oldest first."""
    return _read_raw(ticker, base_dir)["entries"]


def get_latest(ticker: str, base_dir: str = DEFAULT_HISTORY_DIR) -> Optional[dict]:
    """
    Most recent entry BY DATE for one ticker, or None if it has no history
    yet. Because backfilling can insert a past date after later dates
    already exist, "most recent" here is defined as the entry with the
    LATEST Date, not simply the last-written entry -- `entries` is kept
    sorted by Date on every write specifically so "last in the list" and
    "latest by Date" always agree.
    """
    entries = get_history(ticker, base_dir)
    return entries[-1] if entries else None


def get_entry_for_date(ticker: str, entry_date: str, base_dir: str = DEFAULT_HISTORY_DIR) -> Optional[dict]:
    """The entry recorded for an exact Date (ISO "YYYY-MM-DD"), or None if
    nothing was ever saved for that specific date. Used by the Research
    module to detect "you're about to overwrite an existing entry for this
    date" when backfilling."""
    for e in get_history(ticker, base_dir):
        if e.get("Date") == entry_date:
            return e
    return None


def list_tracked_tickers(base_dir: str = DEFAULT_HISTORY_DIR) -> List[str]:
    """Every ticker that has at least one history file on disk."""
    if not os.path.isdir(base_dir):
        return []
    return sorted(fname[:-5] for fname in os.listdir(base_dir) if fname.endswith(".json"))


def days_since_last_update(ticker: str, base_dir: str = DEFAULT_HISTORY_DIR) -> Optional[int]:
    """
    Days since this ticker's most recent recorded entry BY DATE (not by
    wall-clock write time -- see get_latest), or None if it has no history
    at all. Feeds the Research module's freshness dots / the future
    Overview module's "Data Freshness Tracker".
    """
    latest = get_latest(ticker, base_dir)
    if latest is None:
        return None
    try:
        entry_date = datetime.fromisoformat(latest["Date"])
    except (ValueError, TypeError, KeyError):
        return None
    return max((datetime.now() - entry_date).days, 0)


# ---------------------------------------------------------------------------
# CRUD -- per-horizon projection model tuning profiles (Etap 4 follow-up)
# ---------------------------------------------------------------------------

def get_horizon_profile(ticker: str, horizon: str, base_dir: str = DEFAULT_HISTORY_DIR) -> Optional[Dict[str, float]]:
    """
    Returns the saved {"alpha","gamma","kappa","eta"} override for this
    ticker+horizon, or None if nothing has been saved for it -- deliberately
    NOT falling back to engine.single_asset.DEFAULT_HORIZON_PARAMS here
    (this module never imports engine/, per the one-directional
    architecture rule); the UI layer is responsible for substituting the
    engine-level default when this returns None.
    """
    if horizon not in VALID_HORIZONS:
        raise ValueError(f"get_horizon_profile: horizon must be one of {VALID_HORIZONS}, got {horizon!r}")
    profiles = _read_raw(ticker, base_dir)["horizon_profiles"]
    return profiles.get(horizon)


def save_horizon_profile(ticker: str, horizon: str, alpha: float, gamma: float, kappa: float, eta: float, base_dir: str = DEFAULT_HISTORY_DIR) -> Dict[str, float]:
    """Saves/overwrites this ticker's tuning override for one horizon.
    Returns the saved profile dict."""
    if horizon not in VALID_HORIZONS:
        raise ValueError(f"save_horizon_profile: horizon must be one of {VALID_HORIZONS}, got {horizon!r}")
    raw = _read_raw(ticker, base_dir)
    profile = {"alpha": float(alpha), "gamma": float(gamma), "kappa": float(kappa), "eta": float(eta)}
    raw["horizon_profiles"][horizon] = profile
    _write_raw(ticker, raw, base_dir)
    return profile