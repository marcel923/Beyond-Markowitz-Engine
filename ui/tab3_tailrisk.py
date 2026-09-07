"""
ui/tab3_tailrisk.py
=====================
Tab 3: Tail-Risk & Underwater Analytics (Section 4.1) -- per-ticker CDD_0.10 /
Z-score / legacy diagnostic penalty table and underwater chart, plus the
Discrete Crash-Overlap Matrix (Section 3.3) J/K heatmaps and ranking table.

Moved out of quant_terminal.py (Etap 0 architecture split, PROJECT_CONTEXT.md)
with NO behavior change.
"""
import dash
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State

from ui.app_instance import app
from ui.theme import THEME, MATRIX_COLORSCALE, MATRIX_SEQUENTIAL_COLORSCALE
from ui.components import datatable_style_header, datatable_style_cell, datatable_style_data, datatable_row_alt_rule
from engine.risk import compute_drawdown_series, compute_cdd_quantile_vec, compute_crash_overlap_matrix
from engine.returns import compute_composite_upside_row

def compute_tail_risk_metrics(raw_close_data):
    """
    Silnik Tail-Risk (Sekcja 4.1):
      DD_i,t = (P_i,t - running_max(P_i)) / running_max(P_i)            [Underwater series]
      TR_i = CDD_0.10,i = -Quantile_0.10({DD_i,t})                       [Empirical quantile drawdown]
      Z_TR,i = (TR_i - mean(TR)) / std(TR)                               [Z-score przekroju uniwersum]
    Penalty P_i = exp(lambda * max(0, Z_TR,i)) liczone OSOBNO (zależy od lambda, patrz compute_tail_penalty).

    DD/CDD liczone przez `compute_drawdown_series` / `compute_cdd_quantile_vec`
    (tps_solver.py) zamiast duplikować tu tę matematykę -- ta funkcja tylko orkiestruje
    (parsowanie store'a, Z-score, serializacja do JSON dla dcc.Store).

    Zwraca dict: {"tickers":[...], "cdd010":{t:v}, "z":{t:v}, "dd_series":{t:[v,...]}, "dd_dates":{t:[...]}}

    UWAGA (fix): każdy ticker ma TERAZ WŁASNĄ listę dat w "dd_dates", zamiast
    jednej wspólnej listy "dates" branej z pierwszego tickera. Poprzednia wersja
    zakładała identyczny zakres dat dla wszystkich spółek -- fałszywe założenie
    przy uniwersum mieszającym giełdy o różnych kalendarzach sesyjnych (np.
    005930.KS / LYC.AX / ANTO.L obok NYSE/NASDAQ) albo spółki z krótszą
    historią. Przy dużej rozbieżności dat to realnie przesuwało oś X wykresu
    underwater względem wartości DD dla dowolnego tickera innego niż pierwszy.
    """
    prices_df = pd.DataFrame(raw_close_data).set_index('Date')
    prices_df.index = pd.to_datetime(prices_df.index)

    dd_df = compute_drawdown_series(prices_df)
    cdd_vec = compute_cdd_quantile_vec(dd_df, quantile=0.10)

    dd_series, dd_dates, tr_values = {}, {}, {}
    for t in prices_df.columns:
        dd_t = dd_df[t].dropna()
        if len(dd_t) < 2:
            continue
        dd_series[t] = dd_t
        dd_dates[t] = [d.strftime('%Y-%m-%d') for d in dd_t.index]
        tr_values[t] = float(cdd_vec[t])

    valid_tickers = list(tr_values.keys())
    tr_arr = np.array([tr_values[t] for t in valid_tickers])
    mu, sigma = tr_arr.mean(), tr_arr.std(ddof=0)
    z_values = {t: (float((tr_values[t] - mu) / sigma) if sigma > 1e-12 else 0.0) for t in valid_tickers}

    return {
        "tickers": valid_tickers,
        "cdd010": tr_values,
        "z": z_values,
        "dd_series": {t: dd_series[t].round(6).tolist() for t in valid_tickers},
        "dd_dates": dd_dates
    }

def compute_tail_penalty(z_value, lam):
    """P_i = exp(lambda * max(0, Z_TR,i)) — tylko strona downside (Z<0, czyli płytszy niż średni drawdown) nie jest karana."""
    return float(np.exp(lam * max(0.0, z_value)))


@app.callback(Output("panel-stage4a-tailrisk-container", "style"), Input("store-stage3-final-payload", "data"), prevent_initial_call=True)
def reveal_stage4a_tailrisk_panel(payload):
    if payload:
        return {"display": "block", "marginBottom": "35px"}
    return {"display": "none", "marginBottom": "35px"}

@app.callback(Output("panel-crash-overlap-container", "style"), Input("store-stage3-final-payload", "data"), prevent_initial_call=True)
def reveal_crash_overlap_panel(payload):
    if payload:
        return {"display": "block", "marginBottom": "35px"}
    return {"display": "none", "marginBottom": "35px"}


@app.callback(
    Output("store-stage4a-tailrisk", "data"), Output("dropdown-stage4a-stock", "options"), Output("dropdown-stage4a-stock", "value"),
    Input("store-raw-close", "data"), prevent_initial_call=True
)
def compute_stage4a_tailrisk_data(raw_close_data):
    if not raw_close_data:
        return None, [], None
    metrics = compute_tail_risk_metrics(raw_close_data)
    options = [{"label": t, "value": t} for t in metrics["tickers"]]
    default_value = metrics["tickers"][0] if metrics["tickers"] else None
    return metrics, options, default_value

@app.callback(
    Output("store-crash-matrices", "data"),
    Input("store-raw-close", "data"), prevent_initial_call=True
)
def compute_stage4a_crash_matrices(raw_close_data):
    """
    Sekcja 3.3: uruchamia compute_crash_overlap_matrix() (J, S, K) zaraz po
    Stage 1 ingestion, analogicznie do compute_stage4a_tailrisk_data powyżej.
    Osobny callback (nie doklejony do tamtego) mimo wspólnego Inputa -- czytelny
    single-responsibility split, łatwiejszy do debugowania niezależnie od ścieżki
    Z-score/Penalty P_i, która na tym etapie NIE jest jeszcze zasilana przez K
    (patrz PROJECT_CONTEXT.md pkt 6 -- solver dwuetapowy to osobny, kolejny krok).

    DataFrame'y J/S/K serializowane jako {"tickers":[...], "matrix":[[...],...]}
    (lista list w kolejności `tickers`) zamiast `to_dict('records')` używanego
    gdzie indziej w tym pliku -- macierze kwadratowe z tickerami jako ZARÓWNO
    wierszami, jak i kolumnami nie mapują się naturalnie na format "records"
    (nazwy kolumn kolidowałyby z nazwami pól wiersza), więc jawna para
    tickers+matrix jest tu prostsza i jednoznaczna do odtworzenia po stronie
    renderującej.
    """
    if not raw_close_data:
        return None
    prices_df = pd.DataFrame(raw_close_data).set_index('Date')
    prices_df.index = pd.to_datetime(prices_df.index)

    result = compute_crash_overlap_matrix(prices_df, quantile=0.10)
    tickers = list(result["K"].columns)

    return {
        "tickers": tickers,
        "cdd_vec": {t: float(result["cdd_vec"][t]) for t in tickers},
        "J": result["J"].loc[tickers, tickers].round(6).values.tolist(),
        "S": result["S"].loc[tickers, tickers].round(8).values.tolist(),
        "K": result["K"].loc[tickers, tickers].round(8).values.tolist(),
    }

@app.callback(
    Output("graph-crash-jaccard", "figure"), Output("graph-crash-k", "figure"), Output("crash-ranking-table", "children"),
    Input("store-crash-matrices", "data"), prevent_initial_call=True
)
def render_crash_overlap_visuals(crash_data):
    """
    Renderuje Sekcję 3.3: heatmapy J i K (ten sam ciemny styl co heatmapa
    klastrowania w Tab 1 -- patrz `fig_clustered_map` w render_stage2_clustering)
    oraz tabelę rankingową "Crash Risk Contribution" = suma wiersza K per spółka
    (im wyższa, tym bardziej dana spółka podbija zagregowane ryzyko wspólnych
    krachów portfela -- niekoniecznie pokrywa się z samym CDD_0.10, bo K łączy
    głębokość spadku Z siłą nakładania się w czasie z resztą uniwersum).
    """
    empty = go.Figure()
    if not crash_data or not crash_data.get("tickers"):
        return empty, empty, html.Div("Brak danych.", style={"color": THEME["text_dim"], "padding": "10px"})

    tickers = crash_data["tickers"]
    J = np.array(crash_data["J"])
    K = np.array(crash_data["K"])
    cdd_vec = crash_data["cdd_vec"]

    def make_heatmap(matrix, zmax, colorbar_title):
        fig = go.Figure(data=go.Heatmap(
            z=matrix, x=tickers, y=tickers, zmin=0.0, zmax=zmax,
            colorscale=MATRIX_SEQUENTIAL_COLORSCALE,
            text=matrix, texttemplate="%{text:.2f}", textfont=dict(size=10, color=THEME["text_white"]),
            xgap=2, ygap=2,
            hoverongaps=False, colorbar=dict(title=colorbar_title, tickfont=dict(color=THEME["text_dim"]))
        ))
        fig.update_layout(
            template="plotly_dark", height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"],
            margin=dict(l=40, r=40, t=20, b=40), font_family=THEME["font"],
            xaxis=dict(showgrid=False, tickfont=dict(color=THEME["text_white"], size=10)),
            yaxis=dict(showgrid=False, tickfont=dict(color=THEME["text_white"], size=10), autorange="reversed")
        )
        return fig

    fig_j = make_heatmap(J, zmax=1.0, colorbar_title="J")
    k_max = float(K.max()) if K.size and K.max() > 0 else 1.0
    fig_k = make_heatmap(K, zmax=k_max, colorbar_title="K")

    contribution = K.sum(axis=1)  # suma wiersza K, zgodnie ze specyfikacją (K_ii = CDD_i^2 wliczane)
    ranking_df = pd.DataFrame({
        "Ticker": tickers,
        "CDD 0.10": [f"{cdd_vec[t]*100:.1f}%" for t in tickers],
        "Crash Risk Contribution": contribution
    }).sort_values("Crash Risk Contribution", ascending=False)
    ranking_df["Crash Risk Contribution"] = ranking_df["Crash Risk Contribution"].map(lambda v: f"{v:.4f}")

    table = dash_table.DataTable(
        data=ranking_df.to_dict('records'),
        columns=[{"name": c, "id": c} for c in ranking_df.columns],
        style_header=datatable_style_header(),
        style_cell=datatable_style_cell(),
        style_data=datatable_style_data(),
        style_data_conditional=[datatable_row_alt_rule(), {'if': {'column_id': 'Crash Risk Contribution'}, 'color': THEME["warn"], 'fontWeight': 'bold'}],
        style_as_list_view=True
    )

    return fig_j, fig_k, table

LEGACY_DIAGNOSTIC_LAMBDA = 0.60  # Tab 3's P_i = exp(lambda*Z) display is PURELY informational/legacy
# (decision (d) of the True Two-Stage SLSQP refactor: P_i never reaches the solver anymore).
# It intentionally does NOT read input-lambda anymore: that field was rescaled to [0.5, 15.0]
# for the new K-based quadratic penalty exp(lambda*sqrt(w.K.w)), which operates on a totally
# different scale than the old per-asset Z-score exponent. Feeding the new (much larger) lambda
# into the old exp(lambda*Z) formula would blow up to nonsensical numbers (e.g. exp(15*2) at a
# Z=+2 outlier) purely because the two formulas share a variable name, not because that's a
# meaningful value for this legacy diagnostic. Fixed at the pre-refactor default (0.60) instead.


@app.callback(
    Output("stage4a-summary-table", "children"),
    Input("input-alpha", "value"), Input("input-gamma", "value"), Input("input-kappa", "value"), Input("input-nref", "value"),
    Input("store-stage4a-tailrisk", "data"), Input("store-stage3-table", "data"),
    prevent_initial_call=True
)
def render_stage4a_summary_table(alpha, gamma, kappa, n_ref, tailrisk, stage3_rows):
    lam = LEGACY_DIAGNOSTIC_LAMBDA
    if not tailrisk or not tailrisk.get("tickers"):
        return html.Div("Brak danych. Uruchom Stage 1.", style={"color": THEME["orange"], "padding": "20px"})
    alpha = alpha if isinstance(alpha, (int, float)) else 0.5
    gamma = gamma if isinstance(gamma, (int, float)) else 1.50
    kappa = kappa if isinstance(kappa, (int, float)) else 1.00
    n_ref = n_ref if isinstance(n_ref, (int, float)) and n_ref > 0 else 8.0

    stage3_by_ticker = {r["Ticker"]: r for r in (stage3_rows or [])}

    rows = []
    for t in tailrisk["tickers"]:
        z = tailrisk["z"][t]
        p_i = compute_tail_penalty(z, lam)
        s3_row = stage3_by_ticker.get(t, {})
        upside = compute_composite_upside_row(s3_row, gamma, kappa, n_ref, alpha=alpha)
        rows.append({
            "Ticker": t, "CDD 0.10": tailrisk["cdd010"][t], "Z-Score": z, "Penalty (Pi)": p_i,
            "U_component": upside["U_component"], "G_component": upside["G_component"], "mu_i": upside["mu_i"]
        })
    columns = [
        {"name": "Ticker", "id": "Ticker"},
        {"name": "CDD 0.10 (i)", "id": "CDD 0.10", "type": "numeric", "format": {"specifier": ".1%"}},
        {"name": "Z-Score (Z_TR,i)", "id": "Z-Score", "type": "numeric", "format": {"specifier": "+.2f"}},
        {"name": "Penalty (P_i)", "id": "Penalty (Pi)", "type": "numeric", "format": {"specifier": ".2f"}},
        {"name": "U_component (1-α)", "id": "U_component", "type": "numeric", "format": {"specifier": "+.1%"}},
        {"name": "G_component (α)", "id": "G_component", "type": "numeric", "format": {"specifier": "+.1%"}},
        {"name": "μ_i (Composite Upside)", "id": "mu_i", "type": "numeric", "format": {"specifier": "+.1%"}},
    ]
    return dash_table.DataTable(
        columns=columns, data=rows, page_size=15, sort_action='native',
        style_header=datatable_style_header(),
        style_data=datatable_style_data(),
        style_cell=datatable_style_cell(),
        style_cell_conditional=[{'if': {'column_id': 'Ticker'}, 'fontWeight': 'bold', 'textAlign': 'left', 'color': THEME['accent']}],
        style_data_conditional=[
            datatable_row_alt_rule(),
            {'if': {'filter_query': '{Penalty (Pi)} > 1.5', 'column_id': 'Penalty (Pi)'}, 'color': THEME['warn'], 'fontWeight': 'bold'},
            {'if': {'filter_query': '{mu_i} > 0', 'column_id': 'mu_i'}, 'color': THEME['pos']},
            {'if': {'filter_query': '{mu_i} < 0', 'column_id': 'mu_i'}, 'color': THEME['neg']}
        ]
    )

@app.callback(
    Output("graph-stage4a-underwater", "figure"),
    Input("dropdown-stage4a-stock", "value"), State("store-stage4a-tailrisk", "data"),
    prevent_initial_call=True
)
def render_stage4a_underwater_chart(ticker, tailrisk):
    if not ticker or not tailrisk or ticker not in tailrisk.get("dd_series", {}):
        return go.Figure()
    lam = LEGACY_DIAGNOSTIC_LAMBDA
    dates = pd.to_datetime(tailrisk["dd_dates"][ticker])
    dd = np.array(tailrisk["dd_series"][ticker]) * 100.0
    cdd010 = tailrisk["cdd010"][ticker]
    z = tailrisk["z"][ticker]
    p_i = compute_tail_penalty(z, lam)
    threshold_pct = -cdd010 * 100.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=dd, mode='lines', fill='tozeroy', line=dict(color=THEME["accent"], width=1.5),
                              fillcolor="rgba(111,44,255,0.15)", name=f"{ticker} Underwater DD", hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>"))
    fig.add_hline(y=threshold_pct, line_dash="dash", line_color=THEME["orange"], line_width=2,
                  annotation_text=f"10th Percentile Underwater Floor (CDD 0.10) = {threshold_pct:.1f}%",
                  annotation_font_color=THEME["orange"], annotation_position="bottom left")
    fig.update_layout(
        template="plotly_dark", height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"], font_family=THEME["font"],
        title=dict(text=f"{ticker} — Z={z:+.2f}σ, Penalty P_i={p_i:.2f}x", font=dict(color=THEME["text_white"], size=14)),
        showlegend=False, margin=dict(l=50, r=30, t=50, b=40),
        yaxis=dict(ticksuffix="%", gridcolor="#1E1E28", tickfont=dict(color=THEME["text_dim"])),
        xaxis=dict(gridcolor="#1E1E28", tickfont=dict(color=THEME["text_dim"]))
    )
    return fig