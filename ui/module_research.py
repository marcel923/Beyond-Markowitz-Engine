"""
ui/module_research.py
=======================
Module 2: Research / Company Dossier (Etap 4), extended (2026-09-07 follow-up)
into a full Single-Asset Projection & Diagnostic Engine, and further
extended (2026-09-08 follow-up) into a Walk-Forward Calibration & Backfit
Engine: the historical fundamental snapshots already saved for a ticker are
now used to check how well the model's past predictions matched what
actually happened once each horizon elapsed, and (via "AUTODOPASUJ
PARAMETRY") to automatically fit (alpha, gamma, kappa, eta) against that
history instead of relying solely on manual tuning.

Two charts, per the confirmed design:
  Chart A (research-dossier-chart): 5Y price history + saved historical
    Target Consensus/High/Low points + historical model predictions shifted
    to their evaluation date (so you can see "what did the model think a
    year ago about today, vs. what actually happened") + the LIVE forward
    cone, anchored at today's actual market price (NOT at whatever date is
    selected in the backfill DatePicker -- see render_main_chart_and_kpis).
  Chart B (research-backtest-chart): per-entry return-prediction error bars
    (green/red) + a Realization Ratio line (R_real/mu_model), matching the
    same historical entries listed in the Backtest Inspection Table below it.

Three-column IBKR-style layout (search + company list | dossier metrics +
projection chart + KPI strip + backtest chart + backtest table + model
tuning | update form with historical backfilling) wired to
data/universe_store.py, data/company_store.py, and engine/single_asset.py.

Design decisions confirmed with the project owner (2026-09-07/08 conversation,
three passes):
- No valuation ratios (P/E, P/S) snapshot -- yfinance has no historical
  time series for these; dropped.
- New tickers are auto-added to the universe (as "Watchlist") the moment
  "+ SLEDZ" is clicked, with Name/Sector auto-fetched from yfinance --
  tracking a company and entering its fundamental data are two independent
  steps, not gated behind each other.
- Freshness threshold: 30 days fresh / 60 days stale.
- Deliberately NOT wired into the Rebalance workflow's Stage 3 table in
  this pass -- distinct, larger follow-up, explicitly deferred.
- eta ("Execution / Realization Factor"): a manual multiplicative
  correction for a company's historical track record of delivering vs.
  missing guidance. Neutral default 1.0, now also a FITTABLE parameter via
  AUTODOPASUJ. See engine/single_asset.py.
- Historical backfilling: `research-input-date` (DatePickerSingle) lets the
  user save a fundamental snapshot under ANY past date, not just today.
- The forward projection cone (Chart A) ALWAYS anchors at the live/most-recent
  cached market price, decoupled from whatever the backfill DatePicker
  currently has selected -- confirmed correction, 2026-09-08: the form's
  P0 field follows the DatePicker (for saving historical snapshots
  correctly), but the forward-looking cone is a "starting from right now"
  statement and must not silently use a stale backfilled price as if it
  were today's.

No Dash business logic lives outside callbacks in this file beyond small
formatting/lookup helpers -- persistence is entirely data.universe_store /
data.company_store, math is entirely engine.single_asset, kept as thin a
UI layer as the other tab modules (one-directional dependency rule: this
file imports engine/ and data/, neither of those ever imports back).
"""
import json
from datetime import date, timedelta

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State, ALL

from ui.app_instance import app
from ui.theme import THEME
from ui.components import build_kpi_strip, datatable_style_header, datatable_style_cell, datatable_style_data, datatable_row_alt_rule, parse_single_ticker_input
from data import universe_store as uni
from data import company_store as comp
from data.market_data import fetch_current_prices, fetch_company_profile, fetch_universe_prices
from engine.single_asset import (
    compute_single_asset_projection, evaluate_historical_accuracy, fit_single_asset_parameters,
    DEFAULT_HORIZON_PARAMS, MIN_ENTRIES_FOR_FIT,
)

FRESHNESS_FRESH_DAYS = 30
FRESHNESS_STALE_DAYS = 60


def _freshness_dot_color(days):
    """None (no history at all) or > FRESHNESS_STALE_DAYS -> red; between
    FRESH and STALE -> amber; <= FRESH -> green. Matches the original spec's
    "<30 dni" freshness wording, with an intermediate amber band rather
    than a hard binary so a company isn't flagged identically whether it's
    31 days old or 300."""
    if days is None or days > FRESHNESS_STALE_DAYS:
        return THEME["neg"]
    if days > FRESHNESS_FRESH_DAYS:
        return THEME["warn"]
    return THEME["pos"]


def _freshness_label(days):
    return "brak danych" if days is None else f"{days} dni"


def _price_series_from_cache(price_cache):
    """store-research-price-history.data -> pd.Series indexed by date
    (string YYYY-MM-DD), or an empty Series if the cache is empty/missing."""
    if not price_cache or not price_cache.get("dates"):
        return pd.Series(dtype=float)
    return pd.Series(price_cache["prices"], index=price_cache["dates"])


def _lookup_price_asof(price_cache, target_date_str):
    """Nearest close price AT OR BEFORE target_date_str from the cached 5Y
    series -- used when backfilling a past date that may not itself be a
    trading day (weekend/holiday). Returns None if the cache is empty or
    has no observation on/before that date."""
    series = _price_series_from_cache(price_cache)
    if series.empty:
        return None
    eligible = series[series.index <= target_date_str]
    if eligible.empty:
        return None
    return float(eligible.iloc[-1])


# ---------------------------------------------------------------------------
# Lewa kolumna: lista spolek (filtrowana wyszukiwarka)
# ---------------------------------------------------------------------------

@app.callback(
    Output("research-company-list", "children"),
    Input("research-search-input", "value"),
    Input("store-research-refresh", "data"),
    prevent_initial_call=False,
)
def render_company_list(search_value, _refresh):
    companies = uni.list_companies()
    if search_value:
        needle = search_value.strip().upper()
        companies = [c for c in companies if needle in c.get("Ticker", "").upper()]

    if not companies:
        return html.Div("Brak spółek w uniwersum -- dodaj pierwszą poniżej.",
                         style={"padding": "16px", "fontSize": "11px", "color": THEME["text_dim"]})

    rows = []
    for c in companies:
        ticker = c["Ticker"]
        days = comp.days_since_last_update(ticker)
        rows.append(html.Div(
            id={"type": "research-company-row", "ticker": ticker}, n_clicks=0,
            style={"display": "flex", "alignItems": "center", "justifyContent": "space-between",
                   "padding": "9px 16px", "borderBottom": f"1px solid {THEME['border']}", "fontSize": "12.5px", "cursor": "pointer"},
            children=[
                html.Span([
                    html.Span(style={"width": "7px", "height": "7px", "borderRadius": "50%", "display": "inline-block",
                                      "marginRight": "8px", "backgroundColor": _freshness_dot_color(days)}),
                    html.Span(ticker, style={"fontWeight": "600"}),
                ]),
                html.Span(_freshness_label(days), style={"fontSize": "10px", "color": THEME["text_label"]}),
            ]
        ))
    return rows


# ---------------------------------------------------------------------------
# Wybor spolki: klik na wiersz LUB "+ SLEDZ" nowego tickera
# ---------------------------------------------------------------------------

@app.callback(
    Output("store-research-selected-ticker", "data"),
    Output("store-research-refresh", "data", allow_duplicate=True),
    Output("research-new-ticker-status", "children"),
    Input({"type": "research-company-row", "ticker": ALL}, "n_clicks"),
    Input("research-new-ticker-btn", "n_clicks"),
    State("research-new-ticker-input", "value"),
    State("store-research-refresh", "data"),
    prevent_initial_call=True,
)
def select_ticker(_row_clicks, _new_clicks, new_ticker_value, refresh_counter):
    """
    Klik na istniejacy wiersz: tylko wybiera ticker do wyswietlenia.
    "+ SLEDZ" nowego tickera: dodaje go do uniwersum NATYCHMIAST (auto-fetch
    Name/Sector), zanim jakiekolwiek dane fundamentalne zostana wpisane --
    sledzenie i wypelnianie danych to dwa niezalezne kroki.

    Walidacja przez parse_single_ticker_input (ui/components.py, dzielona z
    Tab 1) -- poprawka po realnym bledzie znalezionym w testach: wpisanie
    "AVGO, CRDO" bylo wczesniej cicho akceptowane jako JEDEN literalny
    ticker.
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "research-new-ticker-btn":
        ticker, error = parse_single_ticker_input(new_ticker_value)
        if error:
            return dash.no_update, dash.no_update, html.Div(error, style={"color": THEME["neg"]})
        profile = fetch_company_profile(ticker)
        uni.upsert_company(ticker, name=profile["name"], sector=profile["sector"], status="Watchlist")
        return ticker, (refresh_counter or 0) + 1, html.Div(f"Dodano {ticker}.", style={"color": THEME["pos"]})

    if any(c["value"] for c in ctx.triggered if c["value"]):
        try:
            parsed = json.loads(trigger_id)
            return parsed.get("ticker"), dash.no_update, dash.no_update
        except (json.JSONDecodeError, AttributeError):
            return dash.no_update, dash.no_update, dash.no_update
    return dash.no_update, dash.no_update, dash.no_update


# ---------------------------------------------------------------------------
# Cache 5Y historii cen dla wybranej spolki (jedno pobranie na wybor tickera,
# ponownie uzywane przez: tlo wykresu, sigma/delta_down silnika projekcji,
# oraz odczyt historycznej ceny przy backfillingu z DatePickera).
# ---------------------------------------------------------------------------

@app.callback(
    Output("store-research-price-history", "data"),
    Input("store-research-selected-ticker", "data"),
    prevent_initial_call=True,
)
def cache_price_history(ticker):
    if not ticker:
        return dash.no_update
    prices_df, valid = fetch_universe_prices([ticker], period="5y")
    if prices_df.empty or ticker not in valid:
        return {"dates": [], "prices": []}
    series = prices_df[ticker].dropna()
    return {"dates": [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in series.index], "prices": [float(v) for v in series.values]}


# ---------------------------------------------------------------------------
# Srodek + prawo: wypelnienie dossier i formularza po wyborze spolki
# (P0 i wykres projekcji sa wlasnoscia INNYCH callbackow -- patrz
# sync_p0_with_date i render_projection_chart_and_kpis ponizej -- zeby uniknac
# dwoch callbackow piszacych do tych samych Outputow niezaleznie).
# ---------------------------------------------------------------------------

@app.callback(
    Output("research-dossier-header", "children"),
    Output("research-dossier-metrics", "children"),
    Output("research-input-name", "value"),
    Output("research-input-sector", "value"),
    Output("research-input-status", "value"),
    Output("research-input-target", "value"),
    Output("research-input-thigh", "value"),
    Output("research-input-tlow", "value"),
    Output("research-input-nanalysts", "value"),
    Output("research-input-epscagr", "value"),
    Output("research-input-epsrev", "value"),
    Output("research-input-date", "date"),
    Input("store-research-selected-ticker", "data"),
    prevent_initial_call=True,
)
def populate_dossier(ticker):
    if not ticker:
        return dash.no_update

    company = uni.get_company(ticker)
    latest = comp.get_latest(ticker)

    name = company.get("Name", "") if company else ""
    sector = company.get("Sector", "") if company else ""
    status = company.get("Status", "Watchlist") if company else "Watchlist"

    header_bits = f"{ticker}"
    if name:
        header_bits += f" — {name}"
    if sector:
        header_bits += f" · {sector}"
    header_bits += f" · {status}"

    if latest is None:
        metrics = html.Div("Brak zapisanych wpisów dla tej spółki -- uzupełnij formularz po prawej i zapisz pierwszy snapshot.",
                            style={"fontSize": "12px", "color": THEME["text_dim"]})
        target, thigh, tlow, nanalysts, epscagr, epsrev = None, None, None, None, None, None
    else:
        def metric(label, value):
            return html.Div([
                html.Div(label, style={"fontSize": "10.5px", "color": THEME["text_label"], "marginBottom": "4px"}),
                html.Div(value, style={"fontSize": "15px", "fontWeight": "600", "fontVariantNumeric": "tabular-nums"}),
            ])
        metrics = html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "12px"}, children=[
            metric("Current Price (P0)", f"{latest['P0']:.2f}"),
            metric("Target Consensus", f"{latest['Target_Consensus']:.2f}"),
            metric("Target High / Low", f"{latest['Target_High']:.0f} / {latest['Target_Low']:.0f}"),
            metric("Analyst Coverage", f"{latest['N_analysts']:.0f}"),
            metric("EPS 2Y CAGR", f"{latest['EPS_CAGR']:+.1f}%"),
            metric("90d EPS Revision", f"{latest['EPS_Rev_90d']:+.1f}%"),
        ])
        target, thigh, tlow = latest["Target_Consensus"], latest["Target_High"], latest["Target_Low"]
        nanalysts, epscagr, epsrev = latest["N_analysts"], latest["EPS_CAGR"], latest["EPS_Rev_90d"]

    # Reset daty do dzisiaj przy kazdym nowym wyborze spolki -- to (poprzez
    # lancuch Dasha, patrz sync_p0_with_date ponizej, ktora nasluchuje na
    # zmiane tej daty) rowniez odswieza P0 live dla nowo wybranego tickera,
    # bez potrzeby ustawiania P0 bezposrednio w tym callbacku.
    today_str = date.today().isoformat()

    return header_bits, metrics, name, sector, status, target, thigh, tlow, nanalysts, epscagr, epsrev, today_str


# ---------------------------------------------------------------------------
# P0 podazajacy za wybrana data: dzisiaj -> live fetch; przeszlosc -> odczyt
# z cache 5Y (backfilling historii, 2026-09-07 follow-up).
# ---------------------------------------------------------------------------

@app.callback(
    Output("research-input-p0", "value"),
    Input("research-input-date", "date"),
    State("store-research-selected-ticker", "data"),
    State("store-research-price-history", "data"),
    prevent_initial_call=True,
)
def sync_p0_with_date(selected_date, ticker, price_cache):
    if not ticker or not selected_date:
        return dash.no_update

    if selected_date >= date.today().isoformat():
        live_prices = fetch_current_prices([ticker])
        return live_prices.get(ticker)

    historical_p0 = _lookup_price_asof(price_cache, selected_date)
    return historical_p0  # None jesli cache pusty/brak obserwacji przed ta data -- uzytkownik wpisze recznie


# ---------------------------------------------------------------------------
# Suwaki strojenia: wczytaj zapisany profil dla tickera+horyzontu, albo
# domyslne wartosci silnika, przy kazdej zmianie tickera LUB horyzontu.
# ---------------------------------------------------------------------------

@app.callback(
    Output("research-slider-alpha", "value"),
    Output("research-slider-gamma", "value"),
    Output("research-slider-kappa", "value"),
    Output("research-slider-eta", "value"),
    Output("research-stability-alpha", "children"),
    Output("research-stability-gamma", "children"),
    Output("research-stability-kappa", "children"),
    Output("research-stability-eta", "children"),
    Input("research-horizon-selector", "value"),
    Input("store-research-selected-ticker", "data"),
    prevent_initial_call=True,
)
def load_horizon_sliders(horizon, ticker):
    if not ticker or not horizon:
        return dash.no_update
    saved = comp.get_horizon_profile(ticker, horizon)
    defaults = DEFAULT_HORIZON_PARAMS[horizon]
    p = saved or defaults
    # Plakietki stabilnosci czyscimy przy kazdej zmianie kontekstu -- etykieta
    # z POPRZEDNIEGO dopasowania (innej spolki/horyzontu) nie ma tu zastosowania.
    return p["alpha"], p["gamma"], p["kappa"], p["eta"], "", "", "", ""


# ---------------------------------------------------------------------------
# Zwijalny panel strojenia
# ---------------------------------------------------------------------------

@app.callback(
    Output("research-tuning-panel", "style"),
    Output("research-tuning-toggle-label", "children"),
    Input("research-tuning-toggle", "n_clicks"),
    State("research-tuning-panel", "style"),
    prevent_initial_call=True,
)
def toggle_tuning_panel(_n_clicks, current_style):
    is_open = (current_style or {}).get("display") == "block"
    if is_open:
        return {"display": "none", "marginTop": "14px"}, "▸ rozwiń"
    return {"display": "block", "marginTop": "14px"}, "▾ zwiń"


# ---------------------------------------------------------------------------
# Zapis profilu strojenia dla tego tickera+horyzontu
# ---------------------------------------------------------------------------

@app.callback(
    Output("research-horizon-save-status", "children"),
    Input("research-btn-save-horizon", "n_clicks"),
    State("store-research-selected-ticker", "data"), State("research-horizon-selector", "value"),
    State("research-slider-alpha", "value"), State("research-slider-gamma", "value"),
    State("research-slider-kappa", "value"), State("research-slider-eta", "value"),
    prevent_initial_call=True,
)
def save_horizon_profile_btn(_n_clicks, ticker, horizon, alpha, gamma, kappa, eta):
    if not ticker:
        return html.Div("Wybierz spółkę przed zapisem profilu.", style={"color": THEME["neg"]})
    comp.save_horizon_profile(ticker, horizon, alpha=alpha, gamma=gamma, kappa=kappa, eta=eta)
    return html.Div(f"Zapisano profil {horizon} dla {ticker}.", style={"color": THEME["pos"]})


# ---------------------------------------------------------------------------
# Wykres A: 5Y cena + zapisane punkty Target + historyczne prognozy modelu
# przesuniete na ich date ewaluacji + zywy stozek projekcji. Pasek KPI.
# Stozek ZAWSZE zakotwiczony w NAJSWIEZSZEJ cenie z cache (nie w polu
# formularza P0, ktore podaza za DatePickerem backfillingu -- patrz notatka
# w naglowku modulu).
# ---------------------------------------------------------------------------

@app.callback(
    Output("research-dossier-chart", "figure"),
    Output("research-kpi-strip", "children"),
    Input("store-research-price-history", "data"),
    Input("research-horizon-selector", "value"),
    Input("research-slider-alpha", "value"), Input("research-slider-gamma", "value"),
    Input("research-slider-kappa", "value"), Input("research-slider-eta", "value"),
    Input("research-input-target", "value"),
    Input("research-input-thigh", "value"), Input("research-input-tlow", "value"),
    Input("research-input-nanalysts", "value"), Input("research-input-epscagr", "value"), Input("research-input-epsrev", "value"),
    State("store-research-selected-ticker", "data"),
    prevent_initial_call=True,
)
def render_main_chart_and_kpis(price_cache, horizon, alpha, gamma, kappa, eta,
                                target, thigh, tlow, nanalysts, epscagr, epsrev, ticker):
    fig = go.Figure()

    price_series = _price_series_from_cache(price_cache)
    if not price_series.empty:
        fig.add_trace(go.Scatter(x=list(price_series.index), y=list(price_series.values), name="Cena (5Y)",
                                  line=dict(color=THEME["text_dim"], width=1.3)))

    history = comp.get_history(ticker) if ticker else []
    if history:
        hdates = [h["Date"] for h in history]
        fig.add_trace(go.Scatter(x=hdates, y=[h["Target_Consensus"] for h in history], name="Target Consensus (zapisany)",
                                  mode="markers", marker=dict(color=THEME["pos"], size=6)))
        fig.add_trace(go.Scatter(x=hdates, y=[h["Target_High"] for h in history], name="Target High (zapisany)",
                                  mode="markers", marker=dict(color=THEME["warn"], size=5, symbol="triangle-up")))
        fig.add_trace(go.Scatter(x=hdates, y=[h["Target_Low"] for h in history], name="Target Low (zapisany)",
                                  mode="markers", marker=dict(color=THEME["warn"], size=5, symbol="triangle-down")))

        # Historyczne prognozy modelu, przesuniete na ich WLASNA date ewaluacji
        # (t_k + horyzont sesji) -- "gdzie rok temu model widzial cene na dzisiaj,
        # w porownaniu z tym co sie stalo".
        if horizon and not price_series.empty:
            params = {"alpha": alpha, "gamma": gamma, "kappa": kappa, "eta": eta}
            backtest_rows = evaluate_historical_accuracy(history, price_series, horizon, params)
            if backtest_rows:
                fig.add_trace(go.Scatter(
                    x=[r["date_target"] for r in backtest_rows], y=[r["p_pred"] for r in backtest_rows],
                    name=f"Prognoza modelu (sprzed {horizon})", mode="markers",
                    marker=dict(color=THEME["accent"], size=7, symbol="diamond-open", line=dict(width=1.5)),
                ))

    kpi_children = []
    # Stozek ZAWSZE od najswiezszej ceny w cache -- "dzisiaj", niezaleznie od
    # tego, jaka data jest akurat wybrana w DatePickerze backfillingu.
    live_p0 = float(price_series.iloc[-1]) if not price_series.empty else None
    have_inputs = live_p0 and all(v is not None for v in [target, thigh, tlow, nanalysts])
    if have_inputs and horizon:
        params = {"alpha": alpha, "gamma": gamma, "kappa": kappa, "eta": eta}
        proj = compute_single_asset_projection(
            P0=live_p0, Ti=target, Thigh=thigh, Tlow=tlow, Ni=nanalysts,
            G_2Y=epscagr or 0.0, delta_eps_90d=epsrev or 0.0,
            historical_prices=price_series.values if len(price_series) else [],
            horizon=horizon, params=params,
        )

        today = date.today()
        proj_dates = [(today + timedelta(days=int(t))).isoformat() for t in proj["t_days"]]

        fig.add_trace(go.Scatter(x=proj_dates, y=list(proj["p_upper"]), name="P90 (górna wstęga)",
                                  line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=proj_dates, y=list(proj["p_lower"]), name="Stożek P10–P90", fill="tonexty",
                                  fillcolor="rgba(46,134,255,0.12)", line=dict(width=0), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=proj_dates, y=list(proj["p_proj"]), name=f"Projekcja ({horizon}, od dziś)",
                                  line=dict(color=THEME["accent"], width=2.5)))

        for label, value, color in [("Target Consensus", target, THEME["pos"]), ("Target High", thigh, THEME["warn"]), ("Target Low", tlow, THEME["warn"])]:
            fig.add_hline(y=value, line=dict(color=color, width=1, dash="dot"), annotation_text=label,
                          annotation_font=dict(size=9, color=color), annotation_position="right")

        kpi_children = build_kpi_strip([
            {"label": "Standalone Return μ(h)", "value": f"{proj['mu_h']*100:+.1f}%",
             "color": THEME["pos"] if proj["mu_h"] >= 0 else THEME["neg"]},
            {"label": "Win Probability P(R>0)", "value": f"{proj['p_profit']*100:.1f}%", "color": THEME["accent"]},
            {"label": "Asymmetry Ratio (Up/Down)", "value": f"{proj['asymmetry_ratio']:.2f}"},
            {"label": "Downside Volatility δ_down", "value": f"{proj['delta_down_ann']*100:.1f}%"},
        ])

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"],
        margin=dict(l=40, r=90, t=10, b=30), font_family=THEME["font"],
        legend=dict(orientation="h", y=1.18, font=dict(size=9, color=THEME["text_dim"])),
        xaxis=dict(showgrid=False, tickfont=dict(color=THEME["text_dim"], size=10)),
        yaxis=dict(showgrid=True, gridcolor=THEME["border"], tickfont=dict(color=THEME["text_dim"], size=10)),
    )
    if price_series.empty and not have_inputs:
        fig.add_annotation(text="Wybierz spółkę i uzupełnij dane, żeby zobaczyć projekcję",
                            showarrow=False, font=dict(size=11, color=THEME["text_dim"]), xref="paper", yref="paper", x=0.5, y=0.5)

    return fig, kpi_children


# ---------------------------------------------------------------------------
# Wykres B (Realizacja Prognoz w Czasie) + Tabela Realizacji Historycznej
# (Backtest Inspection Table). Zalezy TYLKO od juz zapisanych wpisow
# historycznych + biezacych suwakow -- NIE od pol formularza "nowego" wpisu
# (te opisuja dane jeszcze niezapisane, backtest patrzy wylacznie wstecz).
# ---------------------------------------------------------------------------

@app.callback(
    Output("research-backtest-chart", "figure"),
    Output("research-backtest-table", "children"),
    Input("store-research-price-history", "data"),
    Input("research-horizon-selector", "value"),
    Input("research-slider-alpha", "value"), Input("research-slider-gamma", "value"),
    Input("research-slider-kappa", "value"), Input("research-slider-eta", "value"),
    State("store-research-selected-ticker", "data"),
    prevent_initial_call=True,
)
def render_backtest_chart_and_table(price_cache, horizon, alpha, gamma, kappa, eta, ticker):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"],
        margin=dict(l=40, r=50, t=10, b=30), font_family=THEME["font"],
        legend=dict(orientation="h", y=1.2, font=dict(size=9, color=THEME["text_dim"])),
        xaxis=dict(showgrid=False, tickfont=dict(color=THEME["text_dim"], size=10)),
        yaxis=dict(title="Błąd zwrotu (p.p.)", showgrid=True, gridcolor=THEME["border"], tickfont=dict(color=THEME["text_dim"], size=10)),
        yaxis2=dict(title="Realization Ratio", overlaying="y", side="right", showgrid=False, tickfont=dict(color=THEME["text_dim"], size=10)),
    )

    if not ticker or not horizon:
        fig.add_annotation(text="Wybierz spółkę, żeby zobaczyć historyczną realizację prognoz",
                            showarrow=False, font=dict(size=11, color=THEME["text_dim"]), xref="paper", yref="paper", x=0.5, y=0.5)
        return fig, html.Div("Brak spółki wybranej.", style={"fontSize": "11px", "color": THEME["text_dim"], "padding": "12px"})

    price_series = _price_series_from_cache(price_cache)
    history = comp.get_history(ticker)
    params = {"alpha": alpha, "gamma": gamma, "kappa": kappa, "eta": eta}
    rows = evaluate_historical_accuracy(history, price_series, horizon, params)

    if not rows:
        fig.add_annotation(text=f"Brak wpisów, dla których horyzont {horizon} już minął -- za mało historii, żeby ocenić realizację",
                            showarrow=False, font=dict(size=11, color=THEME["text_dim"]), xref="paper", yref="paper", x=0.5, y=0.5)
        return fig, html.Div("Brak ewaluowalnych wpisów dla tego horyzontu.", style={"fontSize": "11px", "color": THEME["text_dim"], "padding": "12px"})

    dates_k = [r["date_k"] for r in rows]
    errors_pp = [r["error"] * 100.0 for r in rows]
    bar_colors = [THEME["neg"] if e > 0 else THEME["pos"] for e in errors_pp]  # error>0 = model przestrzelil (za optymistyczny)
    ratios = [r["realization_ratio"] for r in rows]

    fig.add_trace(go.Bar(x=dates_k, y=errors_pp, name="Błąd zwrotu (p.p.)", marker_color=bar_colors, yaxis="y"))
    fig.add_trace(go.Scatter(x=dates_k, y=ratios, name="Realization Ratio", mode="lines+markers",
                              line=dict(color=THEME["accent"], width=2), marker=dict(size=6), yaxis="y2", connectgaps=True))

    # --- Tabela ---
    table_rows = []
    for r in rows:
        if r["realization_ratio"] is None:
            status = "N/A"
        elif r["mu_model"] > 0 and r["r_real"] < 0 or r["mu_model"] < 0 and r["r_real"] > 0:
            # Znak sie odwrocil (np. model przewidzial zysk, wyszla strata) -- procent
            # "realizacji" bylby tu myslacy (np. "-37%"), wiec osobna etykieta zamiast liczby.
            status = "Chybione (przeciwny kierunek)"
        elif r["realization_ratio"] >= 1.0:
            status = f"Dowiezione: {r['realization_ratio']*100:.0f}%"
        else:
            status = f"Niedowiezione: {r['realization_ratio']*100:.0f}%"
        table_rows.append({
            "Data Prognozy": r["date_k"], "Cena Bazowa (P0)": r["p0_k"], "Prognoza μ(h) [%]": r["mu_model"] * 100.0,
            "Cena Docelowa Modelu": r["p_pred"], "Data Ewaluacji": r["date_target"], "Cena Rzeczywista": r["p_real"],
            "Realny Zwrot [%]": r["r_real"] * 100.0, "Błąd Zwrotu [p.p.]": r["error"] * 100.0, "Status": status,
        })

    table = dash_table.DataTable(
        columns=[
            {"name": "Data Prognozy (t_k)", "id": "Data Prognozy"},
            {"name": "Cena Bazowa (P0)", "id": "Cena Bazowa (P0)", "type": "numeric", "format": {"specifier": ".2f"}},
            {"name": "Prognoza μ(h) [%]", "id": "Prognoza μ(h) [%]", "type": "numeric", "format": {"specifier": "+.1f"}},
            {"name": "Cena Docelowa Modelu", "id": "Cena Docelowa Modelu", "type": "numeric", "format": {"specifier": ".2f"}},
            {"name": "Data Ewaluacji (t_target)", "id": "Data Ewaluacji"},
            {"name": "Cena Rzeczywista (P_real)", "id": "Cena Rzeczywista", "type": "numeric", "format": {"specifier": ".2f"}},
            {"name": "Realny Zwrot [%]", "id": "Realny Zwrot [%]", "type": "numeric", "format": {"specifier": "+.1f"}},
            {"name": "Błąd Zwrotu [p.p.]", "id": "Błąd Zwrotu [p.p.]", "type": "numeric", "format": {"specifier": "+.1f"}},
            {"name": "Status / Realizacja", "id": "Status"},
        ],
        data=table_rows, page_size=10, sort_action="native",
        style_header=datatable_style_header(), style_data=datatable_style_data(), style_cell=datatable_style_cell(),
        style_data_conditional=[
            datatable_row_alt_rule(),
            {"if": {"filter_query": "{Błąd Zwrotu [p.p.]} > 0", "column_id": "Błąd Zwrotu [p.p.]"}, "color": THEME["neg"]},
            {"if": {"filter_query": "{Błąd Zwrotu [p.p.]} < 0", "column_id": "Błąd Zwrotu [p.p.]"}, "color": THEME["pos"]},
        ],
    )
    return fig, table


# ---------------------------------------------------------------------------
# Autodopasowanie parametrow (alpha, gamma, kappa, eta) pod historie
# ---------------------------------------------------------------------------

@app.callback(
    Output("research-slider-alpha", "value", allow_duplicate=True),
    Output("research-slider-gamma", "value", allow_duplicate=True),
    Output("research-slider-kappa", "value", allow_duplicate=True),
    Output("research-slider-eta", "value", allow_duplicate=True),
    Output("research-autofit-status", "children"),
    Output("research-stability-alpha", "children", allow_duplicate=True),
    Output("research-stability-gamma", "children", allow_duplicate=True),
    Output("research-stability-kappa", "children", allow_duplicate=True),
    Output("research-stability-eta", "children", allow_duplicate=True),
    Input("research-btn-autofit", "n_clicks"),
    State("store-research-selected-ticker", "data"), State("research-horizon-selector", "value"),
    State("store-research-price-history", "data"),
    prevent_initial_call=True,
)
def run_autofit(_n_clicks, ticker, horizon, price_cache):
    """
    Fituje (alpha, gamma, kappa, eta) przez engine.single_asset.fit_single_asset_parameters
    (scipy.optimize.least_squares) i publikuje, obok samych wartosci, miare
    STABILNOSCI kazdego parametru z osobna -- blad standardowy z macierzy
    Jakobiana w punkcie rozwiazania (asymptotyczne przyblizenie klasyczne dla
    nieliniowych najmniejszych kwadratow: Cov(theta) ~ sigma^2 * (J^T J)^-1),
    a nie tylko ogolne ostrzezenie "malo punktow". Wybor tej metody (a nie
    bootstrap) opisany w engine/single_asset.py -- bootstrap na tej samej,
    malej probce danych nie wykryl realnego niedookreslenia w teście
    kontrolnym, ta metoda wykryla je poprawnie.
    """
    empty4 = (dash.no_update,) * 4

    if not ticker:
        return *empty4, html.Div("Wybierz spółkę przed dopasowaniem.", style={"color": THEME["neg"]}), "", "", "", ""

    history = comp.get_history(ticker)
    price_series = _price_series_from_cache(price_cache)
    result = fit_single_asset_parameters(history, price_series, horizon)

    if result is None:
        return *empty4, html.Div(
            f"Za mało historycznych wpisów, dla których horyzont {horizon} już minął "
            f"(potrzeba min. {MIN_ENTRIES_FOR_FIT}) -- nie da się sensownie dopasować parametrów.",
            style={"color": THEME["neg"]}
        ), "", "", "", ""

    label_colors = {"Stabilne": THEME["pos"], "Umiarkowanie stabilne": THEME["warn"],
                     "Niestabilne": THEME["warn"], "Bardzo niestabilne / niezidentyfikowane": THEME["neg"]}

    def badge(name):
        s = result["stability"][name]
        if s["cv"] is None:
            cv_str = "CV=N/A"
        elif s["cv"] > 10.0:  # >1000% -- pokazanie doslownej, absurdalnie wielkiej liczby
            cv_str = "CV>1000%"  # wyglada jak usterka, nie diagnoza; sam fakt "> 1000%" juz mowi wszystko
        else:
            cv_str = f"CV={s['cv']*100:.0f}%"
        color = label_colors.get(s["label"], THEME["text_dim"])
        return html.Span(f"{s['label']} ({cv_str})", style={"color": color, "fontWeight": "600"})

    condition_note = ""
    if result["condition_number"] > 1e4:
        condition_note = " Uwaga: dopasowanie leży w prawie płaskim kierunku (wysokie uwarunkowanie) -- kilka różnych kombinacji parametrów może pasować niemal równie dobrze."

    status = html.Div(
        f"Dopasowano na {result['n_evaluable']} historycznych wpisach (MSE={result['mse']:.6f}): "
        f"α={result['alpha']:.2f}, γ={result['gamma']:.2f}, κ={result['kappa']:.2f}, η={result['eta']:.2f}.{condition_note}",
        style={"color": THEME["warn"] if condition_note else THEME["pos"]}
    )
    return (result["alpha"], result["gamma"], result["kappa"], result["eta"], status,
            badge("alpha"), badge("gamma"), badge("kappa"), badge("eta"))


# ---------------------------------------------------------------------------
# Zapis snapshotu spolki -- teraz pod DOWOLNA data z DatePickera (backfill),
# nie zawsze "dzisiaj".
# ---------------------------------------------------------------------------

@app.callback(
    Output("research-save-status", "children"),
    Output("store-research-refresh", "data", allow_duplicate=True),
    Input("research-btn-save", "n_clicks"),
    State("store-research-selected-ticker", "data"),
    State("research-input-date", "date"),
    State("research-input-name", "value"), State("research-input-sector", "value"), State("research-input-status", "value"),
    State("research-input-p0", "value"), State("research-input-target", "value"),
    State("research-input-thigh", "value"), State("research-input-tlow", "value"),
    State("research-input-nanalysts", "value"), State("research-input-epscagr", "value"), State("research-input-epsrev", "value"),
    State("store-research-refresh", "data"),
    prevent_initial_call=True,
)
def save_snapshot(n_clicks, ticker, entry_date, name, sector, status, p0, target, thigh, tlow, nanalysts, epscagr, epsrev, refresh_counter):
    if not ticker:
        return html.Div("Wybierz lub wpisz ticker przed zapisem.", style={"color": THEME["neg"]}), dash.no_update

    required = {"Current Price (P0)": p0, "Target Consensus": target, "Target High": thigh, "Target Low": tlow,
                "Analyst Coverage": nanalysts, "EPS 2Y CAGR": epscagr, "90d EPS Revision": epsrev}
    missing = [k for k, v in required.items() if v is None]
    if missing:
        return html.Div(f"Uzupełnij pola: {', '.join(missing)}.", style={"color": THEME["neg"]}), dash.no_update

    entry_date = entry_date or date.today().isoformat()
    already_existed = comp.get_entry_for_date(ticker, entry_date) is not None

    try:
        uni.upsert_company(ticker, name=name or "", sector=sector or "", status=status or "Watchlist")
        comp.append_entry(
            ticker, p0=p0, target_consensus=target, target_high=thigh, target_low=tlow,
            n_analysts=nanalysts, eps_cagr=epscagr, eps_rev_90d=epsrev, entry_date=entry_date,
        )
    except Exception as e:
        return html.Div(f"Błąd zapisu: {e}", style={"color": THEME["neg"]}), dash.no_update

    verb = "Zaktualizowano istniejący wpis" if already_existed else "Zapisano nowy wpis"
    return html.Div(f"{verb} dla {ticker} ({entry_date}).", style={"color": THEME["pos"]}), (refresh_counter or 0) + 1