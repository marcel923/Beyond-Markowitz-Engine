"""
ui/tab2_fundamentals.py
=========================
Tab 2: Stage 3 fundamental inputs table -- baseline cluster assignment from
Stage 2's chosen method, manual overrides, and the CONFIRM & EXPORT step that
freezes the payload consumed by Tab 4's solver.

Moved out of quant_terminal.py (Etap 0 architecture split, PROJECT_CONTEXT.md)
with NO behavior change.
"""
import dash
import numpy as np
import pandas as pd
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from ui.app_instance import app
from ui.theme import THEME, CLUSTER_PALETTE
from ui.components import STAGE3_BASELINE_MODELS, STAGE3_FUNDAMENTAL_COLS
from engine.clustering import compute_semicovariance_matrix, semicov_to_semicorr

def compute_stage3_baseline_clusters(baseline_key, k, monthly_returns_data, daily_returns_data):
    """
    Liczy sugerowane klastry dla jednego z 6 modeli bazowych Stage 3.
    Niezależne od silnika Stage 2 — celowo osobna, prosta implementacja, żeby zmiany
    tutaj nigdy nie mogły wpłynąć na już przetestowany Stage 2.
    """
    if baseline_key not in STAGE3_BASELINE_MODELS:
        raise ValueError(f"Nieznany model bazowy: {baseline_key}")
    data_kind, linkage_method = STAGE3_BASELINE_MODELS[baseline_key]

    if data_kind == "standard":
        if not monthly_returns_data:
            raise ValueError("Brak miesięcznych zwrotów — uruchom ponownie Stage 1.")
        df_returns = pd.DataFrame(monthly_returns_data).set_index('Date')
        corr = df_returns.corr().dropna(how='all', axis=0).dropna(how='all', axis=1)
        tickers = list(corr.columns)
        dist_df = np.sqrt(2 * (1 - corr.clip(-1, 1)))
    else:  # semicov
        if not daily_returns_data:
            raise ValueError("Brak dziennych zwrotów — uruchom ponownie Stage 1.")
        df_returns = pd.DataFrame(daily_returns_data).set_index('Date')
        semicov = compute_semicovariance_matrix(df_returns)
        semicorr = semicov_to_semicorr(semicov).dropna(how='all', axis=0).dropna(how='all', axis=1)
        tickers = list(semicorr.columns)
        dist_df = np.sqrt(2 * (1 - semicorr.clip(-1, 1)))

    Z = linkage(squareform(dist_df.values, checks=False), method=linkage_method)
    labels = fcluster(Z, t=k, criterion='maxclust')
    return {tickers[i]: int(labels[i]) for i in range(len(tickers))}


def build_stage3_initial_rows(tickers, cluster_map, raw_close_data):
    """Buduje wiersze tabeli dla nowego uniwersum tickerów (świeży Stage 1). Fundamenty = twarde 0 (żadnych fallbacków)."""
    last_prices = {}
    if raw_close_data:
        prices_df = pd.DataFrame(raw_close_data).set_index('Date')
        for t in tickers:
            if t in prices_df.columns:
                s = prices_df[t].dropna()
                last_prices[t] = round(float(s.iloc[-1]), 2) if len(s) else 0.0

    rows = []
    for t in tickers:
        rows.append({
            "Ticker": t,
            "Assigned Cluster": cluster_map.get(t, 1),
            "Current Price (P0)": last_prices.get(t, 0.0),
            "Target Consensus (Ti)": 0.0, "Target High (T_high)": 0.0, "Target Low (T_low)": 0.0,
            "Analyst Coverage (Ni)": 0, "EPS 2Y CAGR (Gi,2Y)": 0.0, "90d EPS Revision (ΔEPS90d)": 0.0
        })
    return rows


def build_stage3_summary(rows):
    """Karta podsumowania: liczba aktywnych klastrów + skład każdego z nich."""
    if not rows:
        return html.Div("Brak danych.", style={"color": THEME["text_dim"]})
    grouping = {}
    for r in rows:
        c = r.get("Assigned Cluster", 1)
        grouping.setdefault(c, []).append(r["Ticker"])
    blocks = [html.Div(f"ACTIVE CLUSTERS: {len(grouping)}", style={"fontSize": "13px", "fontWeight": "bold", "color": THEME["text_white"], "marginBottom": "14px"})]
    for c in sorted(grouping.keys()):
        color = CLUSTER_PALETTE.get(((int(c) - 1) % 6) + 1, "#888888") if isinstance(c, (int, float)) else "#888888"
        blocks.append(html.Div(style={"marginBottom": "8px", "display": "flex", "alignItems": "flex-start"}, children=[
            html.Div(style={"width": "12px", "height": "12px", "backgroundColor": color, "borderRadius": "3px", "marginRight": "10px", "marginTop": "3px", "flexShrink": "0"}),
            html.Span([html.Span(f"Cluster {c}: ", style={"fontWeight": "bold", "color": THEME["text_white"]}), html.Span(", ".join(grouping[c]), style={"color": THEME["text_dim"]})])
        ]))
    return html.Div(blocks)

# ============================================================
# STAGE 4A — GLOBAL STRATEGY PARAMETERS & TAIL-RISK ANALYTICS (CDD 0.10)
# ============================================================


@app.callback(Output("panel-stage3-container", "style"), Input("panel-config-stage2-container", "style"), prevent_initial_call=True)
def reveal_stage3_panel(stage2_style):
    if stage2_style and stage2_style.get("display") == "block":
        return {"display": "block", "marginBottom": "35px"}
    return {"display": "none", "marginBottom": "35px"}


@app.callback(
    Output("table-stage3-assets", "data"), Output("store-stage3-last-suggestion", "data"),
    Output("store-stage3-manual-clusters", "data", allow_duplicate=True),
    Input("store-raw-close", "data"), Input("dropdown-baseline-model", "value"),
    Input("btn-stage3-reset-overrides", "n_clicks"),
    State("store-monthly-returns", "data"), State("store-daily-returns", "data"), State("dropdown-k-count", "value"),
    State("table-stage3-assets", "data"), State("store-stage3-manual-clusters", "data"),
    prevent_initial_call=True
)
def sync_stage3_table(raw_close_data, baseline_key, reset_clicks, monthly_returns_data, daily_returns_data, k_count, current_rows, manual_flags):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    manual_flags = manual_flags or {}
    k_safe = int(k_count) if isinstance(k_count, (int, float)) else 3

    if not raw_close_data:
        return dash.no_update, dash.no_update, dash.no_update

    try:
        cluster_map = compute_stage3_baseline_clusters(baseline_key, k_safe, monthly_returns_data, daily_returns_data)
    except ValueError:
        return dash.no_update, dash.no_update, dash.no_update

    tickers = list(cluster_map.keys())

    # NOWE URUCHOMIENIE STAGE 1 -> pełny rebuild uniwersum, reset nadpisań
    if trigger_id == "store-raw-close" or not current_rows:
        rows = build_stage3_initial_rows(tickers, cluster_map, raw_close_data)
        return rows, cluster_map, {}

    # RESET RĘCZNYCH NADPISAŃ -> wszystkie wiersze wracają do sugestii ML, fundamenty zostają
    if trigger_id == "btn-stage3-reset-overrides":
        existing_by_ticker = {r["Ticker"]: r for r in current_rows}
        rows = []
        for t in tickers:
            row = dict(existing_by_ticker.get(t, {"Ticker": t}))
            row["Assigned Cluster"] = cluster_map.get(t, 1)
            for col in STAGE3_FUNDAMENTAL_COLS:
                row.setdefault(col, 0.0)
            rows.append(row)
        return rows, cluster_map, {}

    # ZMIANA MODELU BAZOWEGO -> przelicz sugestię, ale zachowaj ręczne nadpisania i WSZYSTKIE fundamenty
    existing_by_ticker = {r["Ticker"]: r for r in current_rows}
    rows = []
    for t in tickers:
        row = dict(existing_by_ticker.get(t, {"Ticker": t}))
        if not manual_flags.get(t, False):
            row["Assigned Cluster"] = cluster_map.get(t, 1)
        for col in STAGE3_FUNDAMENTAL_COLS:
            row.setdefault(col, 0.0)
        rows.append(row)
    return rows, cluster_map, manual_flags


@app.callback(
    Output("stage3-summary-panel", "children"), Output("store-stage3-table", "data"),
    Output("store-stage3-manual-clusters", "data", allow_duplicate=True),
    Input("table-stage3-assets", "data"),
    State("store-stage3-last-suggestion", "data"), State("store-stage3-manual-clusters", "data"),
    prevent_initial_call=True
)
def track_stage3_edits(rows, last_suggestion, manual_flags):
    if not rows:
        return build_stage3_summary([]), rows, manual_flags or {}
    last_suggestion = last_suggestion or {}
    manual_flags = dict(manual_flags or {})
    for r in rows:
        t = r.get("Ticker")
        suggested = last_suggestion.get(t)
        current = r.get("Assigned Cluster")
        if suggested is not None and current is not None and int(current) != int(suggested):
            manual_flags[t] = True
    return build_stage3_summary(rows), rows, manual_flags


@app.callback(
    Output("store-stage3-final-payload", "data"), Output("stage3-confirm-output", "children"),
    Input("btn-stage3-confirm", "n_clicks"), State("store-stage3-table", "data"),
    prevent_initial_call=True
)
def confirm_stage3_export(n_clicks, table_data):
    if not table_data:
        return dash.no_update, html.Span("Brak danych do eksportu.", style={"color": THEME["orange"]})
    n_assets = len(table_data)
    n_clusters = len(set(r.get("Assigned Cluster") for r in table_data))
    missing_price = [r["Ticker"] for r in table_data if not r.get("Current Price (P0)")]
    warning = f" — Brak ceny dla: {', '.join(missing_price)}" if missing_price else ""
    msg = html.Span(f"Zablokowano {n_assets} aktywów w {n_clusters} klastrach — gotowe dla Stage 4.{warning}",
                     style={"color": THEME["orange"] if missing_price else THEME["accent"], "fontWeight": "bold"})
    return table_data, msg