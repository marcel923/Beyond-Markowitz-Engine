"""
data/market_data.py
====================
All yfinance I/O for this project lives here, and only here. Previously this
logic was duplicated in two places: an inline `yf.download(...)` block inside
the Stage 1 ingestion callback (quant_terminal.py), and two separate
functions in snapshot_store.py for live/forward price lookups. Consolidated
here (Etap 0 architecture split, PROJECT_CONTEXT.md) so there is exactly one
place that knows how to talk to yfinance, one place that handles its
MultiIndex-columns quirk, and one place to swap in a different data vendor
later without hunting through the UI layer for scattered `yf.download` calls.

No Dash code lives here. Callers in ui/ are responsible for turning failures
into user-facing messages -- these functions raise/return empty on failure,
they never touch a dcc.Store or an html.Div.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


def fetch_universe_prices(tickers: List[str], period: str = "5y") -> Tuple[pd.DataFrame, List[str]]:
    """
    Batched historical daily close fetch for Stage 1 ingestion. Handles
    yfinance's MultiIndex-columns quirk with `group_by="ticker"` (which
    triggers even for a single-ticker request), forward/back-fills small
    per-ticker gaps, and reports back which of the requested tickers actually
    resolved to real data.

    Returns (prices_df, valid_tickers). `prices_df` is empty and
    `valid_tickers` is [] on total failure (no exception raised) -- exactly
    the same "empty means failure" contract the rest of this project's data
    functions use, so callers can handle it uniformly.
    """
    import yfinance as yf

    if not tickers:
        return pd.DataFrame(), []

    try:
        df_raw = yf.download(tickers, period=period, group_by="ticker", threads=False, auto_adjust=True)
    except Exception:
        return pd.DataFrame(), []

    if df_raw is None or df_raw.empty:
        return pd.DataFrame(), []

    is_multi_col = isinstance(df_raw.columns, pd.MultiIndex)
    if is_multi_col:
        available_top = set(df_raw.columns.get_level_values(0))
        valid_tickers = [t for t in tickers if t in available_top and not df_raw[t].dropna(how='all').empty]
    else:
        valid_tickers = tickers if not df_raw.empty else []

    if not valid_tickers:
        return pd.DataFrame(), []

    prices = pd.DataFrame(index=df_raw.index)
    for t in valid_tickers:
        if is_multi_col:
            prices[t] = df_raw[t]['Close']
        else:
            prices[t] = df_raw['Close']

    prices = prices.dropna(how='all').ffill().bfill()
    return prices, valid_tickers


def fetch_company_profile(ticker: str) -> Dict[str, str]:
    """
    Best-effort fetch of a company's display name and sector via yfinance's
    `Ticker(ticker).info` -- used by the Research module (Etap 4) to
    auto-fill Name/Sector when a brand-new ticker is added to the universe,
    so the user only has to type the ticker itself, not look up its own
    name and sector by hand.

    Unlike `fetch_current_prices`/`fetch_price_history` above, `.info` is a
    single-ticker call (yfinance has no batched equivalent for company
    profile fields), and it is a genuinely different, heavier endpoint than
    the price-history downloads elsewhere in this module -- slower, and
    more prone to returning a sparse/empty dict for thinly-covered or
    non-US tickers (the project's universe already mixes NYSE/NASDAQ with
    KRX/ASX/LSE names elsewhere, and profile coverage for those is less
    reliable than daily price history). Missing fields degrade to "" rather
    than raising, exactly like every other function in this module -- a
    caller should never crash because Yahoo didn't have a sector for a
    given ticker.

    Returns
    -------
    dict with keys "name" and "sector" (each "" if unavailable). Never
    raises; returns {"name": "", "sector": ""} on any failure (bad ticker,
    no network, rate limit, malformed response).
    """
    import yfinance as yf

    empty = {"name": "", "sector": ""}
    if not ticker or not ticker.strip():
        return empty

    try:
        info = yf.Ticker(ticker.strip()).info
    except Exception:
        return empty

    if not isinstance(info, dict):
        return empty

    name = info.get("longName") or info.get("shortName") or ""
    sector = info.get("sector") or ""
    return {"name": name, "sector": sector}


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