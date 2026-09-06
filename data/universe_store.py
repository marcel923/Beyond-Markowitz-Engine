"""
data/universe_store.py
========================
Persistence layer for the master universe of tracked companies (Etap 1,
PROJECT_CONTEXT.md v2 storage spec). A single JSON file holding the list of
every company the system knows about, independent of whether it currently
has fresh fundamental data or is in an active portfolio -- that's tracked
separately per-ticker in data/company_store.py.

Storage format
--------------
storage/universe.json holding a list of records:
    {
        "Ticker":       "NVDA",
        "Name":         "NVIDIA Corporation",
        "Sector":       "Semiconductors",
        "Status":       "Active_Screened" | "Watchlist",
        "Last_Updated": "2026-09-01"
    }

Same atomic write-to-temp-then-os.replace pattern as data/snapshot_store.py,
for the same reason: never leave a truncated/corrupt file behind if the
process is killed mid-write.

No Dash / UI code lives here on purpose -- this module is pure Python +
stdlib json/os, importable and testable independently of the app (Etap 0
architecture split).
"""

from __future__ import annotations

import json
import os
from datetime import date
from typing import List, Optional

DEFAULT_FILE_PATH = "storage/universe.json"
VALID_STATUSES = ("Active_Screened", "Watchlist")


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


def _write_all(companies: List[dict], file_path: str = DEFAULT_FILE_PATH) -> None:
    dirname = os.path.dirname(file_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    tmp_path = file_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(companies, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, file_path)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def upsert_company(
    ticker: str,
    name: str = "",
    sector: str = "",
    status: str = "Watchlist",
    file_path: str = DEFAULT_FILE_PATH,
) -> dict:
    """
    Adds a new company or updates an existing one (matched by ticker,
    case-insensitive on read but stored upper-cased for consistency with how
    tickers are used everywhere else in this project). `Last_Updated` is
    always stamped to today on upsert -- this is what the future Overview
    module's "Data Freshness Tracker" will read.

    Returns the resulting record.
    """
    ticker = ticker.strip().upper()
    if status not in VALID_STATUSES:
        raise ValueError(f"upsert_company: status must be one of {VALID_STATUSES}, got {status!r}")

    companies = _read_all(file_path)
    today = date.today().isoformat()
    record = {"Ticker": ticker, "Name": name, "Sector": sector, "Status": status, "Last_Updated": today}

    for i, c in enumerate(companies):
        if c.get("Ticker", "").upper() == ticker:
            # Preserve fields the caller didn't pass (empty string means "not specified"),
            # so a status-only update doesn't blank out a previously-set Name/Sector.
            merged = dict(c)
            if name:
                merged["Name"] = name
            if sector:
                merged["Sector"] = sector
            merged["Status"] = status
            merged["Last_Updated"] = today
            companies[i] = merged
            _write_all(companies, file_path)
            return merged

    companies.append(record)
    _write_all(companies, file_path)
    return record


def list_companies(status_filter: Optional[str] = None, file_path: str = DEFAULT_FILE_PATH) -> List[dict]:
    """All tracked companies, sorted by Ticker. `status_filter` restricts to
    one of VALID_STATUSES if given."""
    companies = _read_all(file_path)
    if status_filter is not None:
        companies = [c for c in companies if c.get("Status") == status_filter]
    return sorted(companies, key=lambda c: c.get("Ticker", ""))


def get_company(ticker: str, file_path: str = DEFAULT_FILE_PATH) -> Optional[dict]:
    ticker = ticker.strip().upper()
    for c in _read_all(file_path):
        if c.get("Ticker", "").upper() == ticker:
            return c
    return None


def remove_company(ticker: str, file_path: str = DEFAULT_FILE_PATH) -> bool:
    """Removes a company from the universe list. Does NOT touch its
    data/company_history/{ticker}.json file -- historical fundamental data
    is kept even if a company is dropped from the active universe, since it
    remains valid training material for the future ML roadmap (Etap 6)."""
    ticker = ticker.strip().upper()
    companies = _read_all(file_path)
    new_list = [c for c in companies if c.get("Ticker", "").upper() != ticker]
    if len(new_list) == len(companies):
        return False
    _write_all(new_list, file_path)
    return True
