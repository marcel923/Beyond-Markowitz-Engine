"""
ui/module_relative_value.py
=============================
Module: Relative Value (Etap 6, 2026-09-08 follow-up; extended same day
with interactive pair analysis). Surfaces engine/pairs.py's Engle-Granger
cointegration screener and discrete-event backtest simulator.

Two independent sections:
1. Universe scan (`btn-relval-scan`): scans the FULL universe via
   `engine.pairs.scan_universe_diagnostics` -- shows EVERY pair that passes
   the cheap 2Y-drift pre-filter, including near-misses (e.g. "3 of 4
   gates passed"), not just fully-qualifying pairs, per explicit request.
2. Manual pair analysis (`btn-relval-analyze`): lets the user pick ANY two
   tickers from the universe (not just ones that appeared in the scan
   ranking), see their full gate-by-gate diagnostics, and interactively
   explore a discrete-event threshold-switching backtest
   (`engine.pairs.simulate_pair_strategy`) with live-adjustable entry/exit
   Z-thresholds and favour-weight (e.g. 80/20 -> 85/15) sliders.

Confirmed scope (2026-09-08): this whole module is DIAGNOSTIC. Nothing
here writes into Rebalance's mu_i or Stage 1 SLSQP -- the eventual
theta*tanh(-Z) mu-adjustment mechanism is a separate, later, explicitly
deferred step. The interactive backtest here is a daily-granularity
exploration tool for understanding a pair's behavior, not a preview of
that (monthly-cadence, softer) production mechanism.

No Dash business logic lives outside callbacks in this file beyond small
formatting helpers -- the actual math is entirely engine.pairs, kept as
thin a UI layer as the other modules (one-directional dependency rule:
this file imports engine/ and data/, neither of those ever imports back).
"""
from datetime import date

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html, dash_table
from dash.dependencies import Input, Output, State

from ui.app_instance import app
from ui.theme import THEME
from ui.components import datatable_style_header, datatable_style_cell, datatable_style_data, datatable_row_alt_rule
from data import universe_store as uni
from data.market_data import fetch_universe_prices
from engine.pairs import (
    scan_universe_diagnostics, evaluate_pair_diagnostics, simulate_pair_strategy,
    run_backtest_batch, compute_correlations, run_walk_forward_validation,
    run_theta_trailing_stability, simulate_monthly_theta_curve,
)


# ---------------------------------------------------------------------------
# Sekcja 1: skan calego uniwersum, z widocznoscia near-miss
# ---------------------------------------------------------------------------

def _failed_tickers_note(requested, valid):
    """
    Explicit list of which requested tickers did NOT resolve to real price
    data -- confirmed need (2026-09-08, third follow-up): a ticker silently
    dropping out of `valid_tickers` (e.g. a thinly-covered foreign listing
    like a Korean ".KS" ticker failing on yfinance, or simply not being
    tracked in universe_store at all) was previously invisible -- only the
    COUNT of successfully-fetched tickers was ever shown, never which ones
    were missing, making it impossible to tell "my memory-sector pairs
    aren't in the results" apart from "they failed the gates" vs "they
    never even got price data in the first place."
    """
    missing = sorted(set(requested) - set(valid))
    if not missing:
        return None
    return html.Div(f"Nie udało się pobrać danych dla: {', '.join(missing)} (sprawdź, czy są w uniwersum i czy yfinance ma dla nich dane).",
                     style={"color": THEME["warn"], "marginTop": "4px"})


@app.callback(
    Output("relval-results-table", "children"),
    Output("relval-scan-status", "children"),
    Input("btn-relval-scan", "n_clicks"),
    prevent_initial_call=True,
)
def run_relative_value_scan(n_clicks):
    """
    Fetches 5Y prices for EVERY ticker in universe_store, then runs
    scan_universe_diagnostics (full gate-by-gate diagnostics for every pair
    surviving the cheap drift pre-filter -- including near-misses, not just
    fully-qualifying pairs). Blocking (no background-callback infrastructure
    in this pass, confirmed scope) -- dcc.Loading wrapping this Output
    shows a spinner for the duration, which for a large universe is
    genuinely a multi-minute wait, not a UX bug.
    """
    tickers = [c["Ticker"] for c in uni.list_companies()]
    if len(tickers) < 2:
        return html.Div(), html.Div(
            "Uniwersum ma mniej niż 2 spółki -- dodaj więcej przez Research lub Rebalance, zanim uruchomisz skan.",
            style={"color": THEME["neg"]}
        )

    prices_df, valid_tickers = fetch_universe_prices(tickers, period="5y")
    failed_note = _failed_tickers_note(tickers, valid_tickers)
    if prices_df.empty or len(valid_tickers) < 2:
        return html.Div(), html.Div([
            html.Div("Nie udało się pobrać wystarczających danych cenowych dla uniwersum (sprawdź połączenie z yfinance).", style={"color": THEME["neg"]}),
            failed_note,
        ] if failed_note else [html.Div("Nie udało się pobrać wystarczających danych cenowych dla uniwersum (sprawdź połączenie z yfinance).", style={"color": THEME["neg"]})])

    full_df = scan_universe_diagnostics(prices_df, tickers=valid_tickers)

    if full_df.empty:
        status = html.Div([
            html.Div(
                f"Przeskanowano {len(valid_tickers)} spółek ({date.today().isoformat()}) -- "
                f"żadna para nie przeszła nawet progu dryfu 2Y (35 p.p.).",
                style={"color": THEME["text_dim"]}
            ),
        ] + ([failed_note] if failed_note else []))
        return html.Div(), status

    display_df = full_df.copy()
    display_df["Bramki"] = display_df["Gates Passed"].astype(str) + " / 3"
    display_df["Status"] = display_df["All Passed"].map({True: "✓ Kwalifikuje się", False: "Częściowe"})

    table = dash_table.DataTable(
        columns=[
            {"name": "Spółka A", "id": "Ticker A"}, {"name": "Spółka B", "id": "Ticker B"},
            {"name": "Bramki", "id": "Bramki"}, {"name": "Status", "id": "Status"},
            {"name": "Śr. Rozbieżność 5Y (log)", "id": "Avg Relative Divergence", "type": "numeric", "format": {"specifier": ".3f"}},
            {"name": "p-value", "id": "P-Value", "type": "numeric", "format": {"specifier": ".5f"}},
            {"name": "Half-Life [sesje]", "id": "Half-Life", "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": "Hedge Ratio (γ)", "id": "Hedge Ratio", "type": "numeric", "format": {"specifier": ".3f"}},
        ],
        data=display_df.to_dict("records"), page_size=25, sort_action="native", filter_action="native",
        style_header=datatable_style_header(), style_data=datatable_style_data(), style_cell=datatable_style_cell(),
        style_cell_conditional=[{"if": {"column_id": c}, "fontWeight": "bold", "color": THEME["accent"]} for c in ["Ticker A", "Ticker B"]],
        style_data_conditional=[
            datatable_row_alt_rule(),
            {"if": {"filter_query": "{Status} = 'Częściowe'", "column_id": "Status"}, "color": THEME["warn"]},
            {"if": {"filter_query": "{Status} = '✓ Kwalifikuje się'", "column_id": "Status"}, "color": THEME["pos"], "fontWeight": "bold"},
        ],
    )

    n_qualifying = int(full_df["All Passed"].sum())
    status = html.Div([
        html.Div(
            f"Przeskanowano {len(valid_tickers)} spółek ({date.today().isoformat()}) -- "
            f"{len(full_df)} par przeszło próg dryfu 2Y, z czego {n_qualifying} spełnia wszystkie 4 bramki. "
            f"Reszta pokazana jako near-miss (kolumna \"Bramki\").",
            style={"color": THEME["pos"] if n_qualifying else THEME["warn"]}
        ),
    ] + ([failed_note] if failed_note else []))
    return table, status


# ---------------------------------------------------------------------------
# Sekcja 2: rozwijane listy do recznego wyboru pary
# ---------------------------------------------------------------------------

@app.callback(
    Output("relval-picker-a", "options"),
    Output("relval-picker-b", "options"),
    Input("btn-relval-scan", "n_clicks"),   # odswiez tez po skanie, na wypadek nowo dodanych spolek
    prevent_initial_call=False,
)
def populate_pair_pickers(_n_clicks):
    options = [{"label": c["Ticker"], "value": c["Ticker"]} for c in uni.list_companies()]
    return options, options


# ---------------------------------------------------------------------------
# Sekcja 2: analiza wybranej pary -- pobiera dane TYLKO dla tych 2 tickerow
# (nie calego uniwersum), liczy diagnostyke, cachuje do dalszej interakcji
# suwakami bez ponownego pobierania.
# ---------------------------------------------------------------------------

def _diagnostic_badge(label, value_str, passed):
    color = THEME["pos"] if passed else THEME["neg"]
    icon = "✓" if passed else "✗"
    return html.Div(style={"display": "inline-block", "marginRight": "22px", "marginBottom": "8px"}, children=[
        html.Div(label, style={"fontSize": "10px", "color": THEME["text_label"]}),
        html.Div(f"{icon} {value_str}", style={"fontSize": "13px", "fontWeight": "600", "color": color}),
    ])


@app.callback(
    Output("relval-pair-diagnostics", "children"),
    Output("store-relval-pair-data", "data"),
    Input("btn-relval-analyze", "n_clicks"),
    State("relval-picker-a", "value"), State("relval-picker-b", "value"),
    prevent_initial_call=True,
)
def analyze_pair(_n_clicks, ticker_a, ticker_b):
    if not ticker_a or not ticker_b:
        return html.Div("Wybierz obie spółki.", style={"color": THEME["neg"]}), None
    if ticker_a == ticker_b:
        return html.Div("Wybierz dwie RÓŻNE spółki.", style={"color": THEME["neg"]}), None

    prices_df, valid = fetch_universe_prices([ticker_a, ticker_b], period="5y")
    if prices_df.empty or ticker_a not in valid or ticker_b not in valid:
        return html.Div(f"Nie udało się pobrać danych cenowych dla {ticker_a}/{ticker_b}.", style={"color": THEME["neg"]}), None

    pair_df = prices_df[[ticker_a, ticker_b]].dropna(how="any")
    if len(pair_df) < 534:  # TRADING_DAYS_2Y + 30, ta sama minimalna dlugosc co w skanerze
        return html.Div(f"Za mało wspólnej historii cenowej dla {ticker_a}/{ticker_b} (potrzeba min. ~2Y+).", style={"color": THEME["neg"]}), None

    pa, pb = pair_df[ticker_a], pair_df[ticker_b]
    diag = evaluate_pair_diagnostics(pa, pb)

    badges = html.Div(children=[
        _diagnostic_badge("KOINTEGRACJA (p-value)", f"{diag['p_value']:.5f}", diag["coint_pass"]),
        _diagnostic_badge("HALF-LIFE", f"{diag['half_life']:.1f} sesji" if diag['half_life'] not in (float('inf'),) else "brak powrotu", diag["half_life_pass"]),
        _diagnostic_badge("HEDGE RATIO (γ)", f"{diag['hedge_ratio']:.3f}", diag["hedge_ratio_pass"]),
        html.Div(style={"display": "inline-block", "marginRight": "22px", "marginBottom": "8px"}, children=[
            html.Div("ŚR. ROZBIEŻNOŚĆ 5Y (log, informacyjnie)", style={"fontSize": "10px", "color": THEME["text_label"]}),
            html.Div(f"{diag['avg_relative_divergence']:.3f}", style={"fontSize": "13px", "fontWeight": "600", "color": THEME["text_dim"]}),
        ]),
        html.Div(style={"marginTop": "6px", "fontSize": "12px", "fontWeight": "700",
                         "color": THEME["pos"] if diag["all_passed"] else THEME["warn"]},
                 children=f"{diag['gates_passed']}/3 bramek -- {'kwalifikuje się do dalszej analizy' if diag['all_passed'] else 'NIE kwalifikuje się formalnie, ale można eksplorować poniżej'}"),
    ])

    cached = {
        "ticker_a": ticker_a, "ticker_b": ticker_b, "hedge_ratio": diag["hedge_ratio"],
        "dates": [d.strftime("%Y-%m-%d") for d in pair_df.index],
        "prices_a": pa.tolist(), "prices_b": pb.tolist(),
    }
    return badges, cached


# ---------------------------------------------------------------------------
# Wykres 3-panelowy: krzywe kapitalowe / Z-score / struktura wag. Reaguje na
# kazda zmiane progu na zywo, bez ponownego pobierania cen (korzysta z cache
# zapisanego przez analyze_pair).
# ---------------------------------------------------------------------------

@app.callback(
    Output("relval-strategy-chart", "figure"),
    Input("store-relval-pair-data", "data"),
    Input("relval-slider-entry-z", "value"), Input("relval-slider-exit-z", "value"),
    Input("relval-slider-favour-weight", "value"),
    prevent_initial_call=True,
)
def render_strategy_chart(pair_data, entry_z, exit_z, favour_weight):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=["Wyniki kapitałowe (Baza = 100)", "Z-Score spreadu", "Struktura wag w portfelu"],
    )
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"],
        margin=dict(l=50, r=30, t=40, b=30), font_family=THEME["font"],
        legend=dict(orientation="h", y=1.08, font=dict(size=9, color=THEME["text_dim"])),
        height=700,
    )
    for r in (1, 2, 3):
        fig.update_xaxes(showgrid=False, tickfont=dict(color=THEME["text_dim"], size=10), row=r, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=THEME["border"], tickfont=dict(color=THEME["text_dim"], size=10), row=r, col=1)

    if not pair_data:
        fig.add_annotation(text="Wybierz parę i kliknij ANALIZUJ PARĘ", showarrow=False,
                            font=dict(size=12, color=THEME["text_dim"]), xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    dates = pair_data["dates"]
    pa = pd.Series(pair_data["prices_a"], index=pd.to_datetime(dates))
    pb = pd.Series(pair_data["prices_b"], index=pd.to_datetime(dates))
    ta, tb = pair_data["ticker_a"], pair_data["ticker_b"]

    result = simulate_pair_strategy(pa, pb, hedge_ratio=pair_data["hedge_ratio"], entry_z=entry_z, exit_z=exit_z, favour_weight=favour_weight)

    if not result["dates"]:
        fig.add_annotation(text="Za mało danych do policzenia rolling Z-score przy tym oknie.", showarrow=False,
                            font=dict(size=12, color=THEME["text_dim"]), xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    d = result["dates"]
    fig.add_trace(go.Scatter(x=d, y=result["equity_strategy"], name=f"Dynamiczny Long-Only ({round(favour_weight*100)}/{round((1-favour_weight)*100)})",
                              line=dict(color=THEME["pos"], width=2.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d, y=result["equity_benchmark"], name="Pasywny Koszyk 50/50",
                              line=dict(color=THEME["text_white"], width=1.5, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=d, y=result["equity_100a"], name=f"100% {ta}",
                              line=dict(color=THEME["warn"], width=1.2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=d, y=result["equity_100b"], name=f"100% {tb}",
                              line=dict(color=THEME["accent"], width=1.2, dash="dot")), row=1, col=1)

    fig.add_trace(go.Scatter(x=d, y=result["zscore"], name="Z-Score", line=dict(color=THEME["accent"], width=1.3), showlegend=False), row=2, col=1)
    fig.add_hline(y=entry_z, row=2, col=1, line=dict(color=THEME["neg"], dash="dot"), annotation_text=f"Przeważenie {tb}")
    fig.add_hline(y=-entry_z, row=2, col=1, line=dict(color=THEME["pos"], dash="dot"), annotation_text=f"Przeważenie {ta}")
    fig.add_hline(y=0.0, row=2, col=1, line=dict(color=THEME["border_strong"], dash="dash"))

    weight_a_pct = [w * 100 for w in result["weight_a"]]
    weight_b_pct = [100 - w for w in weight_a_pct]
    fig.add_trace(go.Scatter(x=d, y=weight_a_pct, name=f"{ta} Waga %", line=dict(color=THEME["warn"], width=1.0), stackgroup="one", showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=d, y=weight_b_pct, name=f"{tb} Waga %", line=dict(color=THEME["accent"], width=1.0), stackgroup="one", showlegend=False), row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    return fig


# ---------------------------------------------------------------------------
# Wykres theta dla TEJ SAMEJ pary -- mechanizm miesieczny, nie progowy.
# Confirmed 2026-09-08 (osmy follow-up): zbiorczy wykres w zakladce
# "Trwalosc Miesieczna" wygladal jak czysty szum na setkach par naraz --
# ta wersja pozwala zobaczyc DOKLADNIE JEDNA pare, zeby zbudowac intuicje,
# zanim wracamy do pytania czy 12 miesiecy to dobra dlugosc okna.
# ---------------------------------------------------------------------------

@app.callback(
    Output("relval-theta-chart", "figure"),
    Input("store-relval-pair-data", "data"),
    Input("relval-theta-input", "value"), Input("relval-theta-nmonths", "value"),
    prevent_initial_call=True,
)
def render_theta_chart(pair_data, theta, n_months):
    theta = theta or 0.15
    n_months = int(n_months) if n_months else 12

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=["Wyniki kapitałowe (Baza = 100)", "Waga w portfelu (skok raz na miesiąc)"],
    )
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"],
        margin=dict(l=50, r=30, t=40, b=30), font_family=THEME["font"],
        legend=dict(orientation="h", y=1.08, font=dict(size=9, color=THEME["text_dim"])),
        height=620,
    )
    for r in (1, 2):
        fig.update_xaxes(showgrid=False, tickfont=dict(color=THEME["text_dim"], size=10), row=r, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=THEME["border"], tickfont=dict(color=THEME["text_dim"], size=10), row=r, col=1)

    if not pair_data:
        fig.add_annotation(text="Wybierz parę i kliknij ANALIZUJ PARĘ", showarrow=False,
                            font=dict(size=12, color=THEME["text_dim"]), xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    dates = pair_data["dates"]
    pa = pd.Series(pair_data["prices_a"], index=pd.to_datetime(dates))
    pb = pd.Series(pair_data["prices_b"], index=pd.to_datetime(dates))
    ta, tb = pair_data["ticker_a"], pair_data["ticker_b"]

    curve = simulate_monthly_theta_curve(pa, pb, theta=theta, n_months=n_months)
    if not curve["dates"]:
        fig.add_annotation(text="Za mało danych na tyle miesięcy (potrzeba 5 lat treningu + tyle miesięcy testu).", showarrow=False,
                            font=dict(size=12, color=THEME["text_dim"]), xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    d = curve["dates"]
    fig.add_trace(go.Scatter(x=d, y=curve["equity_strategy"], name=f"Theta Nudge (θ={theta})",
                              line=dict(color=THEME["pos"], width=2.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d, y=curve["equity_benchmark"], name="Pasywny Koszyk 50/50",
                              line=dict(color=THEME["text_white"], width=1.5, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=d, y=curve["equity_100a"], name=f"100% {ta}",
                              line=dict(color=THEME["warn"], width=1.2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=d, y=curve["equity_100b"], name=f"100% {tb}",
                              line=dict(color=THEME["accent"], width=1.2, dash="dot")), row=1, col=1)

    for mb in curve["month_boundaries"]:
        fig.add_vline(x=mb["date"], line=dict(color=THEME["border_strong"], width=0.6, dash="dot"), row=1, col=1)

    weight_a_pct = [w * 100 for w in curve["weight_a"]]
    weight_b_pct = [100 - w for w in weight_a_pct]
    fig.add_trace(go.Scatter(x=d, y=weight_a_pct, name=f"{ta} Waga %", line=dict(color=THEME["warn"], width=1.2), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=d, y=weight_b_pct, name=f"{tb} Waga %", line=dict(color=THEME["accent"], width=1.2), showlegend=False), row=2, col=1)
    fig.add_hline(y=50, row=2, col=1, line=dict(color=THEME["border_strong"], dash="dash"))
    fig.update_yaxes(range=[30, 70], row=2, col=1)

    return fig


# ---------------------------------------------------------------------------
# Sekcja 3 (druga zakladka modulu): Backtest Attribution -- pelny backtest
# dla KAZDEJ pary z ostatniego skanu (nie tylko 4/4), ranking po
# RZECZYWISTYM wyniku (Alpha vs benchmark), plus analiza korelacji: co
# faktycznie tlumaczy przewage (nie tylko liczba bramek).
# ---------------------------------------------------------------------------

CORRELATION_CANDIDATE_COLS = [
    "P-Value", "Half-Life", "Hedge Ratio", "Avg Relative Divergence",
    "Z-Score Std", "Raw Spread Volatility", "N Transitions", "Gates Passed", "Same Sector",
]


def _correlation_bar(label, value):
    """Pozioma plakietka korelacji -- dlugosc/kolor wedlug sily i kierunku."""
    color = THEME["pos"] if value > 0 else THEME["neg"]
    width_pct = min(abs(value), 1.0) * 100
    return html.Div(style={"marginBottom": "10px"}, children=[
        html.Div(style={"display": "flex", "justifyContent": "space-between", "fontSize": "11px", "marginBottom": "3px"}, children=[
            html.Span(label, style={"color": THEME["text_dim"]}),
            html.Span(f"{value:+.3f}", style={"color": color, "fontWeight": "700", "fontVariantNumeric": "tabular-nums"}),
        ]),
        html.Div(style={"height": "6px", "backgroundColor": THEME["bg_input"], "borderRadius": "3px", "position": "relative"}, children=[
            html.Div(style={"height": "100%", "width": f"{width_pct}%", "backgroundColor": color, "borderRadius": "3px"}),
        ]),
    ])


@app.callback(
    Output("relval-batch-table", "children"),
    Output("relval-batch-correlations", "children"),
    Output("relval-batch-scatter", "figure"),
    Output("relval-batch-status", "children"),
    Input("btn-relval-batch-run", "n_clicks"),
    State("relval-batch-min-gates", "value"),
    State("relval-slider-entry-z", "value"), State("relval-slider-exit-z", "value"), State("relval-slider-favour-weight", "value"),
    prevent_initial_call=True,
)
def run_batch_attribution(_n_clicks, min_gates, entry_z, exit_z, favour_weight):
    """
    Self-contained (re-fetches + re-scans the universe rather than reusing
    a cached scan -- simpler and avoids a large cross-callback price-data
    cache, at the cost of redoing work the "SKANUJ UNIWERSUM" button in the
    other sub-tab already did; acceptable since this is already a multi-minute
    operation regardless of the extra re-scan cost). Runs
    engine.pairs.run_backtest_batch on every pair with >= min_gates gates
    passed, attaches a "Same Sector" boolean from data.universe_store
    (deliberately computed HERE, not inside engine/pairs.py, since engine/
    never imports data/ per the one-directional dependency rule), then
    computes correlations of every candidate variable against the
    resulting Alpha -- the direct, numeric answer to "what explains the
    outperformance" (confirmed motivation, 2026-09-08 second follow-up).

    Also reports which requested universe tickers failed to fetch at all
    (see _failed_tickers_note) -- confirmed need after the project owner
    could not tell whether specific tickers (memory-sector names) were
    silently missing data vs. genuinely failing the gates.
    """
    entry_z = entry_z or 1.5
    exit_z = exit_z if exit_z is not None else 0.25
    favour_weight = favour_weight or 0.80
    empty_fig = go.Figure()

    tickers = [c["Ticker"] for c in uni.list_companies()]
    if len(tickers) < 2:
        return html.Div(), html.Div(), empty_fig, html.Div("Uniwersum ma mniej niż 2 spółki.", style={"color": THEME["neg"]})

    prices_df, valid_tickers = fetch_universe_prices(tickers, period="5y")
    failed_note = _failed_tickers_note(tickers, valid_tickers)
    if prices_df.empty or len(valid_tickers) < 2:
        msgs = [html.Div("Nie udało się pobrać danych cenowych dla uniwersum.", style={"color": THEME["neg"]})] + ([failed_note] if failed_note else [])
        return html.Div(), html.Div(), empty_fig, html.Div(msgs)

    full_scan = scan_universe_diagnostics(prices_df, tickers=valid_tickers)
    candidate_pairs = full_scan[full_scan["Gates Passed"] >= min_gates]
    if candidate_pairs.empty:
        msgs = [html.Div(f"Żadna para nie ma co najmniej {min_gates}/4 bramek -- obniż próg lub przeskanuj ponownie.", style={"color": THEME["warn"]})] + ([failed_note] if failed_note else [])
        return html.Div(), html.Div(), empty_fig, html.Div(msgs)

    batch = run_backtest_batch(prices_df, candidate_pairs, entry_z=entry_z, exit_z=exit_z, favour_weight=favour_weight)
    if batch.empty:
        msgs = [html.Div("Żadnej pary nie udało się przebacktestować (za mało wspólnej historii).", style={"color": THEME["neg"]})] + ([failed_note] if failed_note else [])
        return html.Div(), html.Div(), empty_fig, html.Div(msgs)

    # "Same Sector" -- doklejane TUTAJ (warstwa UI), nie w engine/pairs.py
    sector_by_ticker = {c["Ticker"]: c.get("Sector", "") for c in uni.list_companies()}
    batch["Same Sector"] = batch.apply(
        lambda r: int(bool(sector_by_ticker.get(r["Ticker A"])) and sector_by_ticker.get(r["Ticker A"]) == sector_by_ticker.get(r["Ticker B"])),
        axis=1
    )

    correlations = compute_correlations(batch, target_col="Relative Alpha [%]", candidate_cols=CORRELATION_CANDIDATE_COLS)
    if correlations.empty:
        corr_display = html.Div("Za mało zróżnicowanych par, żeby policzyć sensowną korelację.", style={"fontSize": "12px", "color": THEME["text_dim"]})
    else:
        corr_display = html.Div([_correlation_bar(name, val) for name, val in correlations.items()])
        corr_display = html.Div([
            html.Div(
                "Dodatnia korelacja = wyższa wartość tej zmiennej idzie w parze z wyższą Alpha. Ujemna = odwrotnie. "
                "To pokazuje, co NAPRAWDĘ tłumaczy wynik -- niekoniecznie to samo, co decyduje o formalnym zakwalifikowaniu pary.",
                style={"fontSize": "11px", "color": THEME["text_dim"], "marginBottom": "14px", "fontStyle": "italic"}
            ),
            corr_display,
        ])

    # --- Wizualizacja: p-value vs Alpha, kolor = ten sam sektor ---
    scatter_fig = go.Figure()
    for same_sector, label, color in [(1, "Ten sam sektor", THEME["accent"]), (0, "Różny sektor", THEME["text_dim"])]:
        subset = batch[batch["Same Sector"] == same_sector]
        if subset.empty:
            continue
        scatter_fig.add_trace(go.Scatter(
            x=subset["P-Value"], y=subset["Relative Alpha [%]"], mode="markers", name=label,
            marker=dict(size=10, color=color, line=dict(width=1, color=THEME["bg_base"])),
            text=[f"{a}/{b}" for a, b in zip(subset["Ticker A"], subset["Ticker B"])],
            hovertemplate="%{text}<br>p-value=%{x:.4f}<br>Alpha=%{y:+.1f}%<extra></extra>",
        ))
    scatter_fig.add_vline(x=0.05, line=dict(color=THEME["warn"], dash="dot"), annotation_text="próg p=0.05")
    scatter_fig.add_hline(y=0.0, line=dict(color=THEME["border_strong"], dash="dash"))
    scatter_fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"],
        margin=dict(l=50, r=30, t=20, b=40), font_family=THEME["font"],
        legend=dict(orientation="h", y=1.1, font=dict(size=10, color=THEME["text_dim"])),
        xaxis=dict(title="p-value (skala log -- niżej = silniejszy formalny dowód kointegracji)", type="log",
                   showgrid=False, tickfont=dict(color=THEME["text_dim"], size=10)),
        yaxis=dict(title="Relative Alpha vs najlepszej alternatywy [%]", showgrid=True, gridcolor=THEME["border"], tickfont=dict(color=THEME["text_dim"], size=10)),
    )

    display_df = batch.copy()
    display_df["Sektor"] = display_df["Same Sector"].map({1: "Ten sam", 0: "Różny"})
    display_df["Bramki"] = display_df["Gates Passed"].astype(str) + " / 3"
    table = dash_table.DataTable(
        columns=[
            {"name": "Spółka A", "id": "Ticker A"}, {"name": "Spółka B", "id": "Ticker B"},
            {"name": "Composite Score", "id": "Composite Score", "type": "numeric", "format": {"specifier": ".3f"}},
            {"name": "Relative Alpha [%]", "id": "Relative Alpha [%]", "type": "numeric", "format": {"specifier": "+.1f"}},
            {"name": "Dni na prowadzeniu [%]", "id": "Days In Lead [%]", "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": "Kapitał Końcowy", "id": "Final Equity", "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": "Max Drawdown [%]", "id": "Max Drawdown [%]", "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": "Bramki", "id": "Bramki"},
            {"name": "Sektor", "id": "Sektor"},
            {"name": "p-value", "id": "P-Value", "type": "numeric", "format": {"specifier": ".5f"}},
            {"name": "Half-Life", "id": "Half-Life", "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": "Przejścia", "id": "N Transitions"},
        ],
        data=display_df.to_dict("records"), page_size=25, sort_action="native", filter_action="native",
        style_header=datatable_style_header(), style_data=datatable_style_data(), style_cell=datatable_style_cell(),
        style_cell_conditional=[{"if": {"column_id": c}, "fontWeight": "bold", "color": THEME["accent"]} for c in ["Ticker A", "Ticker B"]],
        style_data_conditional=[
            datatable_row_alt_rule(),
            {"if": {"filter_query": "{Relative Alpha [%]} > 0", "column_id": "Relative Alpha [%]"}, "color": THEME["pos"], "fontWeight": "bold"},
            {"if": {"filter_query": "{Relative Alpha [%]} < 0", "column_id": "Relative Alpha [%]"}, "color": THEME["neg"]},
        ],
    )

    status = html.Div([
        html.Div(
            f"Przebacktestowano {len(batch)} par (min. {min_gates}/3 bramek, progi: entry={entry_z}, exit={exit_z}, favour={favour_weight}) "
            f"-- najlepsza wg Composite Score: {batch.iloc[0]['Ticker A']}/{batch.iloc[0]['Ticker B']} "
            f"(Relative Alpha {batch.iloc[0]['Relative Alpha [%]']:+.1f}%, na prowadzeniu {batch.iloc[0]['Days In Lead [%]']:.1f}% dni).",
            style={"color": THEME["pos"]}
        ),
    ] + ([failed_note] if failed_note else []))
    return table, corr_display, scatter_fig, status


# ---------------------------------------------------------------------------
# Sekcja 4 (trzecia zakladka modulu): Walk-Forward Out-of-Sample Validation --
# sprawdza, czy przewaga wykryta w danych treningowych utrzymuje sie w
# swiezym, nietknietym roku, ktory nigdy nie wplywal na dobor hedge ratio
# ani bramek tej pary.
# ---------------------------------------------------------------------------

@app.callback(
    Output("relval-oos-table", "children"),
    Output("relval-oos-status", "children"),
    Input("btn-relval-oos-run", "n_clicks"),
    State("relval-oos-min-gates", "value"), State("relval-oos-test-years", "value"),
    State("relval-slider-entry-z", "value"), State("relval-slider-exit-z", "value"), State("relval-slider-favour-weight", "value"),
    prevent_initial_call=True,
)
def run_oos_validation(_n_clicks, min_gates, test_years, entry_z, exit_z, favour_weight):
    """
    Self-contained (fetches 6Y of universe data fresh -- 5Y train + 1Y test
    by default), matching the same pattern as the other two sub-tabs.
    Candidate pairs come from a scan on the FULL 6Y window (filtered by
    min_gates) purely to decide WHICH pairs are worth walk-forward testing
    at all -- the actual hedge ratio / gate diagnostics used for the
    validation itself are recomputed from TRAIN data only, inside
    engine.pairs.run_walk_forward_validation (see that function's
    docstring for why re-using a diagnostic computed on the full window,
    including the test year, would defeat the purpose).
    """
    entry_z = entry_z or 1.5
    exit_z = exit_z if exit_z is not None else 0.25
    favour_weight = favour_weight or 0.80
    test_years = test_years or 1.0

    tickers = [c["Ticker"] for c in uni.list_companies()]
    if len(tickers) < 2:
        return html.Div(), html.Div("Uniwersum ma mniej niż 2 spółki.", style={"color": THEME["neg"]})

    prices_df, valid_tickers = fetch_universe_prices(tickers, period="6y")
    failed_note = _failed_tickers_note(tickers, valid_tickers)
    if prices_df.empty or len(valid_tickers) < 2:
        msgs = [html.Div("Nie udało się pobrać danych cenowych dla uniwersum.", style={"color": THEME["neg"]})] + ([failed_note] if failed_note else [])
        return html.Div(), html.Div(msgs)

    full_scan = scan_universe_diagnostics(prices_df, tickers=valid_tickers)
    candidate_pairs = full_scan[full_scan["Gates Passed"] >= min_gates]
    if candidate_pairs.empty:
        msgs = [html.Div(f"Żadna para nie ma co najmniej {min_gates}/3 bramek na pełnym 6-letnim oknie.", style={"color": THEME["warn"]})] + ([failed_note] if failed_note else [])
        return html.Div(), html.Div(msgs)

    validation = run_walk_forward_validation(prices_df, candidate_pairs, test_years=test_years,
                                              entry_z=entry_z, exit_z=exit_z, favour_weight=favour_weight)
    if validation.empty:
        msgs = [html.Div("Żadna para nie miała wystarczająco danych treningowych i testowych (potrzeba min. ~2Y treningu + ~1 kwartał testu).", style={"color": THEME["neg"]})] + ([failed_note] if failed_note else [])
        return html.Div(), html.Div(msgs)

    table = dash_table.DataTable(
        columns=[
            {"name": "Spółka A", "id": "Ticker A"}, {"name": "Spółka B", "id": "Ticker B"},
            {"name": "Δ Composite (OOS-IS)", "id": "Composite Score Δ (OOS - IS)", "type": "numeric", "format": {"specifier": "+.3f"}},
            {"name": "Composite (IS)", "id": "Composite Score (IS)", "type": "numeric", "format": {"specifier": ".3f"}},
            {"name": "Composite (OOS)", "id": "Composite Score (OOS)", "type": "numeric", "format": {"specifier": ".3f"}},
            {"name": "Alpha IS [%]", "id": "Relative Alpha IS [%]", "type": "numeric", "format": {"specifier": "+.1f"}},
            {"name": "Alpha OOS [%]", "id": "Relative Alpha OOS [%]", "type": "numeric", "format": {"specifier": "+.1f"}},
            {"name": "Na prowadz. IS [%]", "id": "Days In Lead IS [%]", "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": "Na prowadz. OOS [%]", "id": "Days In Lead OOS [%]", "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": "Bramki (Trening)", "id": "Gates Passed (Train)"},
            {"name": "Bramki (OOS, 0-2)", "id": "Gates Passed (OOS)"},
            {"name": "p-value (Trening)", "id": "P-Value (Train)", "type": "numeric", "format": {"specifier": ".5f"}},
            {"name": "p-value (OOS)", "id": "P-Value (OOS)", "type": "numeric", "format": {"specifier": ".5f"}},
            {"name": "Hedge Ratio", "id": "Hedge Ratio", "type": "numeric", "format": {"specifier": ".3f"}},
        ],
        data=validation.to_dict("records"), page_size=25, sort_action="native", filter_action="native",
        style_header=datatable_style_header(), style_data=datatable_style_data(), style_cell=datatable_style_cell(),
        style_cell_conditional=[{"if": {"column_id": c}, "fontWeight": "bold", "color": THEME["accent"]} for c in ["Ticker A", "Ticker B"]],
        style_data_conditional=[
            datatable_row_alt_rule(),
            {"if": {"filter_query": "{Composite Score Δ (OOS - IS)} > 0", "column_id": "Composite Score Δ (OOS - IS)"}, "color": THEME["pos"], "fontWeight": "bold"},
            {"if": {"filter_query": "{Composite Score Δ (OOS - IS)} < 0", "column_id": "Composite Score Δ (OOS - IS)"}, "color": THEME["neg"]},
        ],
    )

    n_improved = int((validation["Composite Score Δ (OOS - IS)"] > 0).sum())
    status = html.Div([
        html.Div(
            f"Zwalidowano {len(validation)} par (trening={test_years} {'rok' if test_years==1 else 'lat'} przed końcem danych) -- "
            f"{n_improved} z {len(validation)} par utrzymało lub poprawiło swój Composite Score poza próbą. "
            f"Najlepsza OOS: {validation.iloc[0]['Ticker A']}/{validation.iloc[0]['Ticker B']}.",
            style={"color": THEME["pos"]}
        ),
    ] + ([failed_note] if failed_note else []))
    return table, status


# ---------------------------------------------------------------------------
# Sekcja 5 (czwarta zakladka modulu): IS Score (5-letni trening, metoda
# progowa z zakladki OOS) x Trailing Score theta (12 miesiecy, ranking
# wzgledem innych par w kazdym miesiacu osobno) -- confirmed design
# (2026-09-08, siodmy follow-up): p-value NIE jest tu uzywane WCALE.
# ---------------------------------------------------------------------------

STABILITY_THRESHOLD = 0.8  # confirmed empirycznie przez wlasciciela projektu


@app.callback(
    Output("relval-monthly-scatter", "figure"),
    Output("relval-monthly-table", "children"),
    Output("relval-monthly-status", "children"),
    Input("btn-relval-monthly-run", "n_clicks"),
    State("relval-monthly-min-gates", "value"), State("relval-monthly-theta", "value"),
    prevent_initial_call=True,
)
def run_monthly_persistence_test(_n_clicks, min_gates, theta):
    """
    Self-contained, fetches 6Y fresh (5Y train + up to 12 test months),
    matching the pattern of the other three sub-tabs. Candidate pairs come
    from a full-window scan filtered by min_gates -- purely a coarse
    initial filter; cointegration p-value plays NO further role anywhere
    in this tab (confirmed design, 2026-09-08 seventh follow-up: "nie wiem
    czy p-value jest tu w ogole przydatne... ciagle gdzies je wrzucasz" --
    it is deliberately absent from both the chart and the table here).

    Combines two DIFFERENT mechanisms by design, made explicit rather than
    silently mixed: "IS Score" reuses run_walk_forward_validation's
    Composite Score (IS) -- the discrete threshold-switching backtest over
    the full 5-year training window (no extra data/compute needed, already
    well-defined). "Trailing Score" is NEW, using the theta/monthly
    production-mechanism prototype specifically, over the trailing 12
    months (run_theta_trailing_stability) -- confirmed instruction ("teraz
    tylko trzeba to zaimplementowac dla strategii z theta"). The two
    axes therefore answer two different but complementary questions: "was
    this pair generally good historically" (IS, any reasonable method) and
    "does the ACTUAL theta mechanism show a stable, consistently
    well-ranked edge recently" (Trailing, the production-candidate method).

    The chart directly visualizes the project owner's own empirical
    observation from an earlier IS/OOS table: pairs scoring above
    STABILITY_THRESHOLD (0.8) in BOTH dimensions tended to beat the best
    single-stock alternative -- rendered here as reference lines on both
    axes plus a distinct marker color for pairs clearing both.
    """
    theta = theta or 0.15

    tickers = [c["Ticker"] for c in uni.list_companies()]
    empty_fig = go.Figure()
    if len(tickers) < 2:
        return empty_fig, html.Div(), html.Div("Uniwersum ma mniej niż 2 spółki.", style={"color": THEME["neg"]})

    prices_df, valid_tickers = fetch_universe_prices(tickers, period="6y")
    failed_note = _failed_tickers_note(tickers, valid_tickers)
    if prices_df.empty or len(valid_tickers) < 2:
        msgs = [html.Div("Nie udało się pobrać danych cenowych dla uniwersum.", style={"color": THEME["neg"]})] + ([failed_note] if failed_note else [])
        return empty_fig, html.Div(), html.Div(msgs)

    full_scan = scan_universe_diagnostics(prices_df, tickers=valid_tickers)
    candidate_pairs = full_scan[full_scan["Gates Passed"] >= min_gates]
    if candidate_pairs.empty:
        msgs = [html.Div(f"Żadna para nie ma co najmniej {min_gates}/3 bramek.", style={"color": THEME["warn"]})] + ([failed_note] if failed_note else [])
        return empty_fig, html.Div(), html.Div(msgs)

    is_scores = run_walk_forward_validation(prices_df, candidate_pairs, test_years=1)
    trailing_scores = run_theta_trailing_stability(prices_df, candidate_pairs, theta=theta)

    if is_scores.empty or trailing_scores.empty:
        msgs = [html.Div("Za mało wspólnej historii, żeby policzyć zarówno IS Score, jak i Trailing Score dla którejkolwiek pary.", style={"color": THEME["neg"]})] + ([failed_note] if failed_note else [])
        return empty_fig, html.Div(), html.Div(msgs)

    merged = pd.merge(
        is_scores[["Ticker A", "Ticker B", "Composite Score (IS)"]],
        trailing_scores[["Ticker A", "Ticker B", "Trailing Score (mean)", "Trailing Score (std)", "N Miesięcy"]],
        on=["Ticker A", "Ticker B"], how="inner",
    )
    if merged.empty:
        msgs = [html.Div("Żadna para nie ma jednocześnie policzonego IS Score i Trailing Score.", style={"color": THEME["neg"]})] + ([failed_note] if failed_note else [])
        return empty_fig, html.Div(), html.Div(msgs)

    merged["Oba > 0.8"] = (merged["Composite Score (IS)"] > STABILITY_THRESHOLD) & (merged["Trailing Score (mean)"] > STABILITY_THRESHOLD)
    merged = merged.sort_values("Trailing Score (mean)", ascending=False).reset_index(drop=True)

    fig = go.Figure()
    for both_high, label, color in [(True, f"Oba > {STABILITY_THRESHOLD}", THEME["pos"]), (False, "Poniżej progu w co najmniej jednym oknie", THEME["text_dim"])]:
        subset = merged[merged["Oba > 0.8"] == both_high]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset["Composite Score (IS)"], y=subset["Trailing Score (mean)"], mode="markers", name=label,
            error_y=dict(type="data", array=subset["Trailing Score (std)"], color=color, thickness=1.3, width=4),
            marker=dict(size=11, color=color, line=dict(width=1, color=THEME["bg_base"])),
            text=[f"{a}/{b}" for a, b in zip(subset["Ticker A"], subset["Ticker B"])],
            hovertemplate="%{text}<br>IS Score=%{x:.3f}<br>Trailing Score=%{y:.3f}<extra></extra>",
        ))
    fig.add_vline(x=STABILITY_THRESHOLD, line=dict(color=THEME["warn"], dash="dot"))
    fig.add_hline(y=STABILITY_THRESHOLD, line=dict(color=THEME["warn"], dash="dot"))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"],
        margin=dict(l=60, r=30, t=20, b=50), font_family=THEME["font"],
        legend=dict(orientation="h", y=1.08, font=dict(size=10, color=THEME["text_dim"])),
        xaxis=dict(title="IS Score (trening 5-letni, metoda progowa)", range=[0, 1.05],
                   showgrid=True, gridcolor=THEME["border"], tickfont=dict(color=THEME["text_dim"], size=10)),
        yaxis=dict(title="Trailing Score (śr. z 12 mies., metoda theta) ± odchylenie std", range=[0, 1.05],
                   showgrid=True, gridcolor=THEME["border"], tickfont=dict(color=THEME["text_dim"], size=10)),
    )

    table = dash_table.DataTable(
        columns=[
            {"name": "Spółka A", "id": "Ticker A"}, {"name": "Spółka B", "id": "Ticker B"},
            {"name": "IS Score", "id": "Composite Score (IS)", "type": "numeric", "format": {"specifier": ".3f"}},
            {"name": "Trailing Score (śr.)", "id": "Trailing Score (mean)", "type": "numeric", "format": {"specifier": ".3f"}},
            {"name": "Trailing Score (std)", "id": "Trailing Score (std)", "type": "numeric", "format": {"specifier": ".3f"}},
            {"name": "N Miesięcy", "id": "N Miesięcy"},
        ],
        data=merged.to_dict("records"), page_size=25, sort_action="native", filter_action="native",
        style_header=datatable_style_header(), style_data=datatable_style_data(), style_cell=datatable_style_cell(),
        style_cell_conditional=[{"if": {"column_id": c}, "fontWeight": "bold", "color": THEME["accent"]} for c in ["Ticker A", "Ticker B"]],
        style_data_conditional=[
            datatable_row_alt_rule(),
            {"if": {"filter_query": f"{{Composite Score (IS)}} > {STABILITY_THRESHOLD} && {{Trailing Score (mean)}} > {STABILITY_THRESHOLD}"}, "backgroundColor": "rgba(0,200,83,0.08)"},
        ],
    )

    n_both_high = int(merged["Oba > 0.8"].sum())
    status = html.Div([
        html.Div(
            f"Policzono {len(merged)} par (theta={theta}, IS=5 lat treningu, Trailing=12 miesięcy) -- "
            f"{n_both_high} z {len(merged)} par ma Score > {STABILITY_THRESHOLD} w OBU oknach czasowych.",
            style={"color": THEME["pos"] if n_both_high else THEME["text_dim"]}
        ),
    ] + ([failed_note] if failed_note else []))
    return fig, table, status