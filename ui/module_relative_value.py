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
    run_backtest_batch, compute_correlations,
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