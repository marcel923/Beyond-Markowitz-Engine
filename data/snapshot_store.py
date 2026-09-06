"""
data/snapshot_store.py
========================
Persistence layer for saved portfolio snapshots ("Forward-Testing / Out-of-Sample
Logging System"). Pure Python -- no Dash code lives here on purpose. All
yfinance calls delegate to data.market_data (Etap 0 architecture split).

Etap 1 storage migration: previously a single "saved_portfolios.json" array
file holding every snapshot. Migrated to ONE JSON FILE PER SNAPSHOT under
storage/portfolio_snapshots/ (PROJECT_CONTEXT.md v2 storage spec) -- avoids
a single growing file being the one thing that can corrupt or merge-conflict
as snapshots accumulate, and makes each snapshot individually
diffable/inspectable on disk. See scripts/migrate_snapshots_etap1.py for the
one-time migration out of the old saved_portfolios.json.

Storage format
--------------
storage/portfolio_snapshots/portfolio_{snapshot_id}.json, one JSON OBJECT
(not a list) per file, shaped like:

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

`storage_dir` parameter naming note: the public CRUD functions below use the
same PARAMETER POSITION/keyword-default pattern as before the migration (a
single optional path override, defaulting to the standard location), just
renamed from `file_path` to `storage_dir` since it now points at a directory,
not a single file. No caller in ui/tab4_rebalance.py or ui/tab5_sandbox.py
passes this argument explicitly (verified before this migration), so the
rename does not break anything.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from data.market_data import fetch_current_prices, fetch_price_history

DEFAULT_STORAGE_DIR = "storage/portfolio_snapshots"
BENCHMARK_TICKER = "SPY"


# ---------------------------------------------------------------------------
# Low-level file I/O
# ---------------------------------------------------------------------------

def _snapshot_file_path(snapshot_id: str, storage_dir: str = DEFAULT_STORAGE_DIR) -> str:
    """Sanitizes snapshot_id for use as a filename -- a no-op for the normal
    "YYYY-MM-DD_HH-MM[_N]" format (already filesystem-safe), but a defensive
    guard against anything unexpected ever ending up as a snapshot_id."""
    safe_id = re.sub(r'[^A-Za-z0-9_\-]', '_', snapshot_id)
    return os.path.join(storage_dir, f"portfolio_{safe_id}.json")


def _read_one(snapshot_id: str, storage_dir: str = DEFAULT_STORAGE_DIR) -> Optional[dict]:
    path = _snapshot_file_path(snapshot_id, storage_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)
        return record if isinstance(record, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _read_all(storage_dir: str = DEFAULT_STORAGE_DIR) -> List[dict]:
    """Reads every snapshot file in storage_dir. Malformed individual files
    are silently skipped (never crash the whole listing over one bad file) --
    same defensive philosophy as the pre-migration single-file reader."""
    if not os.path.isdir(storage_dir):
        return []
    snapshots = []
    for fname in os.listdir(storage_dir):
        if not (fname.startswith("portfolio_") and fname.endswith(".json")):
            continue
        try:
            with open(os.path.join(storage_dir, fname), "r", encoding="utf-8") as f:
                record = json.load(f)
            if isinstance(record, dict):
                snapshots.append(record)
        except (json.JSONDecodeError, OSError):
            continue
    return snapshots


def _write_one(record: dict, storage_dir: str = DEFAULT_STORAGE_DIR) -> None:
    os.makedirs(storage_dir, exist_ok=True)
    path = _snapshot_file_path(record["snapshot_id"], storage_dir)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


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
    storage_dir: str = DEFAULT_STORAGE_DIR,
) -> dict:
    """
    Writes a new snapshot as its own file and returns the created record.

    `z_scores` (optional): per-ticker Z_TR,i tail-risk Z-scores frozen at save time
    (Section 4.1). These are lambda-INDEPENDENT, so storing them lets a later "what-if"
    sandbox recompute P_i = exp(new_lambda * max(0, Z)) correctly for any lambda the user
    tries -- without needing to re-download 5 years of price history. Old snapshots saved
    before this field existed simply won't have it; callers should treat a missing/absent
    z_scores dict as "no frozen risk data available" and degrade gracefully (e.g. assume
    Z=0 / P_i=1 baseline), not crash.
    """
    base_id = make_snapshot_id()
    existing_ids = {s.get("snapshot_id") for s in _read_all(storage_dir)}
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
    _write_one(record, storage_dir)
    return record


def list_snapshots(storage_dir: str = DEFAULT_STORAGE_DIR) -> List[dict]:
    """Newest first -- most useful ordering for a dropdown."""
    snapshots = _read_all(storage_dir)
    return sorted(snapshots, key=lambda s: s.get("created_at", ""), reverse=True)


def get_snapshot(snapshot_id: str, storage_dir: str = DEFAULT_STORAGE_DIR) -> Optional[dict]:
    return _read_one(snapshot_id, storage_dir)


def delete_snapshot(snapshot_id: str, storage_dir: str = DEFAULT_STORAGE_DIR) -> bool:
    path = _snapshot_file_path(snapshot_id, storage_dir)
    if not os.path.exists(path):
        return False
    os.remove(path)
    return True


def rename_snapshot(snapshot_id: str, new_name: str, storage_dir: str = DEFAULT_STORAGE_DIR) -> bool:
    if not new_name or not new_name.strip():
        return False
    record = _read_one(snapshot_id, storage_dir)
    if record is None:
        return False
    record["snapshot_name"] = new_name.strip()
    _write_one(record, storage_dir)
    return True


# ---------------------------------------------------------------------------
# Live price lookups + forward-return math
# ---------------------------------------------------------------------------

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