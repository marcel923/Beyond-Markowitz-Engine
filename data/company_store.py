"""
data/company_store.py
=======================
Persistence layer for per-company historical fundamental data (Etap 1,
PROJECT_CONTEXT.md v2 storage spec). One JSON file per ticker under
storage/company_history/, holding an append-only log of fundamental
snapshots over time -- this is the raw material both for the future Company
Dossier module (target-price-vs-actual-price chart) and, much later, for
training an ML expected-return model (Etap 6 roadmap): the whole reason this
gets its own store now, well before any ML work starts, is that history
only accumulates from whenever recording starts. There is no way to
backfill it later.

Storage format
--------------
storage/company_history/{TICKER}.json holding a list of records, oldest first:
    {
        "Date":             "2026-09-01",
        "P0":               217.44,
        "Target_Consensus": 300.0,
        "Target_High":      400.0,
        "Target_Low":       100.0,
        "N_analysts":       12,
        "EPS_CAGR":         12.0,
        "EPS_Rev_90d":      0.0
    }

One entry per Date: re-recording the same Date UPDATES that entry in place
rather than appending a duplicate (matches the spec's "wpisy z każdego
miesiąca" cadence -- if the Dossier's update form gets used twice in the
same day, that's a correction, not two separate historical data points).

Same atomic write-to-temp-then-os.replace pattern as the other data/ stores.
No Dash / UI code lives here on purpose.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime
from typing import List, Optional

DEFAULT_HISTORY_DIR = "storage/company_history"


# ---------------------------------------------------------------------------
# Low-level file I/O
# ---------------------------------------------------------------------------

def _ticker_file_path(ticker: str, base_dir: str = DEFAULT_HISTORY_DIR) -> str:
    """Sanitizes the ticker for use as a filename. Most tickers (NVDA,
    005930.KS, LYC.AX) are already filesystem-safe as-is; this only guards
    against the rare ticker convention that includes a path separator."""
    safe_ticker = re.sub(r'[\\/]', '_', ticker.strip().upper())
    return os.path.join(base_dir, f"{safe_ticker}.json")


def _read_history(ticker: str, base_dir: str = DEFAULT_HISTORY_DIR) -> List[dict]:
    path = _ticker_file_path(ticker, base_dir)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _write_history(ticker: str, entries: List[dict], base_dir: str = DEFAULT_HISTORY_DIR) -> None:
    os.makedirs(base_dir, exist_ok=True)
    path = _ticker_file_path(ticker, base_dir)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# CRUD
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
    Records a fundamental data point for `ticker`. `entry_date` defaults to
    today (ISO format) if not given. If an entry for that exact date already
    exists, it is REPLACED (not duplicated) -- see module docstring.

    Returns the resulting entry.
    """
    entry_date = entry_date or date.today().isoformat()
    entry = {
        "Date": entry_date, "P0": float(p0), "Target_Consensus": float(target_consensus),
        "Target_High": float(target_high), "Target_Low": float(target_low),
        "N_analysts": int(n_analysts), "EPS_CAGR": float(eps_cagr), "EPS_Rev_90d": float(eps_rev_90d),
    }

    entries = _read_history(ticker, base_dir)
    for i, e in enumerate(entries):
        if e.get("Date") == entry_date:
            entries[i] = entry
            entries.sort(key=lambda e: e.get("Date", ""))
            _write_history(ticker, entries, base_dir)
            return entry

    entries.append(entry)
    entries.sort(key=lambda e: e.get("Date", ""))
    _write_history(ticker, entries, base_dir)
    return entry


def get_history(ticker: str, base_dir: str = DEFAULT_HISTORY_DIR) -> List[dict]:
    """Full history for one ticker, oldest first."""
    return _read_history(ticker, base_dir)


def get_latest(ticker: str, base_dir: str = DEFAULT_HISTORY_DIR) -> Optional[dict]:
    """Most recent entry for one ticker, or None if it has no history yet."""
    entries = _read_history(ticker, base_dir)
    return entries[-1] if entries else None


def list_tracked_tickers(base_dir: str = DEFAULT_HISTORY_DIR) -> List[str]:
    """Every ticker that has at least one history file on disk."""
    if not os.path.isdir(base_dir):
        return []
    return sorted(fname[:-5] for fname in os.listdir(base_dir) if fname.endswith(".json"))


def days_since_last_update(ticker: str, base_dir: str = DEFAULT_HISTORY_DIR) -> Optional[int]:
    """
    Days since this ticker's most recent recorded entry, or None if it has
    no history at all. Feeds the future Overview module's "Data Freshness
    Tracker" (companies with data <30 days old vs needing an update).
    """
    latest = get_latest(ticker, base_dir)
    if latest is None:
        return None
    try:
        entry_date = datetime.fromisoformat(latest["Date"])
    except (ValueError, TypeError, KeyError):
        return None
    return max((datetime.now() - entry_date).days, 0)
