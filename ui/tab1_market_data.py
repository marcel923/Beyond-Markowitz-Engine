"""
ui/tab1_market_data.py
========================
Tab 1: ticker ingestion (Stage 1) and unsupervised clustering (Stage 2) --
Ward/Complete/Single/Average linkage, Estrada semi-covariance, RMT denoising,
DTW + K-Medoids, DBSCAN -- plus the raw-data inspector sidebar and the
multi-asset price chart.

Moved out of quant_terminal.py (Etap 0 architecture split, PROJECT_CONTEXT.md)
with NO behavior change.
"""
import dash
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram, fcluster
from scipy.spatial.distance import squareform

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ui.app_instance import app
from ui.theme import THEME, CLUSTER_PALETTE, CHART_COLORS, MATRIX_COLORSCALE, MATRIX_SEQUENTIAL_COLORSCALE
from ui.components import generate_tws_matrix_styles, datatable_style_header, datatable_style_cell, datatable_style_data, datatable_row_alt_rule, parse_single_ticker_input
from data import universe_store as uni
from data.market_data import fetch_company_profile
from engine.clustering import (
    compute_semicovariance_matrix, semicov_to_semicorr, rmt_denoise_correlation,
    compute_elbow_eps, dtw_distance, compute_dtw_distance_matrix, kmedoids,
)

def build_dendrogram_figure(Z, active_tickers, ticker_to_cluster):
    dendro_data = dendrogram(Z, labels=active_tickers, no_plot=True)
    fig_dendro = go.Figure()
    leaf_positions = np.arange(5, len(active_tickers) * 10, 10)
    for xs, ys in zip(dendro_data['icoord'], dendro_data['dcoord']):
        left_leaf_idx = int((xs[0] - 5) / 10) if (xs[0] - 5) % 10 == 0 else None
        right_leaf_idx = int((xs[3] - 5) / 10) if (xs[3] - 5) % 10 == 0 else None
        branch_color = "#444454"
        if left_leaf_idx is not None and left_leaf_idx < len(dendro_data['ivl']):
            branch_color = CLUSTER_PALETTE.get(ticker_to_cluster[dendro_data['ivl'][left_leaf_idx]], branch_color)
        elif right_leaf_idx is not None and right_leaf_idx < len(dendro_data['ivl']):
            branch_color = CLUSTER_PALETTE.get(ticker_to_cluster[dendro_data['ivl'][right_leaf_idx]], branch_color)
        fig_dendro.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color=branch_color, width=2.5), showlegend=False, hoverinfo='none'))
    fig_dendro.update_layout(
        template="plotly_dark", height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"],
        margin=dict(l=50, r=30, t=10, b=40), font_family=THEME["font"],
        xaxis=dict(tickmode='array', tickvals=leaf_positions, ticktext=dendro_data['ivl'], tickfont=dict(color=THEME["text_white"], size=12), showgrid=False),
        yaxis=dict(gridcolor="#1E1E28", tickfont=dict(color=THEME["text_dim"], size=10), title="Linkage Distance")
    )
    return fig_dendro

def build_cluster_legend(ticker_to_cluster, model_label):
    blocks = [html.Div(f"RISK MODEL: {model_label}", style={"fontSize": "11px", "color": THEME["text_dim"], "fontWeight": "bold", "marginBottom": "12px", "letterSpacing": "0.5px"})]
    unique_labels = sorted(set(ticker_to_cluster.values()), key=lambda x: (x == -1, x))
    color_idx = 1
    for lab in unique_labels:
        members = [t for t, c in ticker_to_cluster.items() if c == lab]
        if lab == -1:
            color, title = "#555566", "NOISE / DIVERSIFIER"
        else:
            color, title = CLUSTER_PALETTE.get(color_idx, "#FFFFFF"), f"CLUSTER 0{color_idx}"
            color_idx += 1
        blocks.append(html.Div(style={"marginBottom": "8px", "display": "flex", "alignItems": "center"}, children=[
            html.Div(style={"width": "14px", "height": "14px", "backgroundColor": color, "borderRadius": "4px", "marginRight": "12px"}),
            html.Span(f"{title}: ", style={"fontWeight": "bold", "color": THEME["text_white"], "marginRight": "8px"}),
            html.Span(", ".join(members), style={"color": THEME["text_dim"]})
        ]))
    return blocks

HIERARCHICAL_LINKAGE = {"ward": "ward", "complete": "complete", "single": "single", "average": "average"}
HIERARCHICAL_LABEL = {"ward": "Ward Linkage", "complete": "Complete Linkage", "single": "Single Linkage", "average": "Average Linkage"}

def get_method_distance_matrix(method, monthly_returns_data, daily_returns_data, raw_close_data):
    """
    JEDNO ŹRÓDŁO PRAWDY dla przygotowania danych każdej metody — używane zarówno przez
    silnik klastrowania (run_stage_02_clustering) jak i sugestię optymalnego k (silhouette),
    żeby obie ścieżki zawsze liczyły na dokładnie tej samej macierzy odległości.
    Zwraca: (dist_df, tickers, kind, linkage_method_or_None, extra)
      kind = "hierarchical" -> Ward/Complete/Single/Average/Semicov/RMT (linkage + fcluster)
      kind = "partition"    -> DTW + K-Medoids (brak natywnego dendrogramu)
      extra = dict z dodatkowymi danymi do etykiet (np. RMT eigenvalues)
    Rzuca ValueError z czytelnym komunikatem gdy brak danych lub metoda nie dotyczy (DBSCAN).
    """
    if method in HIERARCHICAL_LINKAGE:
        if not monthly_returns_data:
            raise ValueError("Brak miesięcznych zwrotów — uruchom ponownie Stage 1.")
        df_returns = pd.DataFrame(monthly_returns_data).set_index('Date')
        corr = df_returns.corr().dropna(how='all', axis=0).dropna(how='all', axis=1)
        tickers = list(corr.columns)
        dist_df = np.sqrt(2 * (1 - corr.clip(-1, 1)))
        return dist_df, tickers, "hierarchical", HIERARCHICAL_LINKAGE[method], {}

    elif method == "semicov":
        if not daily_returns_data:
            raise ValueError("Brak dziennych zwrotów — uruchom ponownie Stage 1.")
        df_returns = pd.DataFrame(daily_returns_data).set_index('Date')
        semicov = compute_semicovariance_matrix(df_returns)
        semicorr = semicov_to_semicorr(semicov).dropna(how='all', axis=0).dropna(how='all', axis=1)
        tickers = list(semicorr.columns)
        dist_df = np.sqrt(2 * (1 - semicorr.clip(-1, 1)))
        return dist_df, tickers, "hierarchical", "ward", {}

    elif method == "rmt_denoised":
        if not daily_returns_data:
            raise ValueError("Brak dziennych zwrotów — uruchom ponownie Stage 1.")
        df_returns = pd.DataFrame(daily_returns_data).set_index('Date')
        denoised, lam_plus, eigvals = rmt_denoise_correlation(df_returns)
        denoised = denoised.dropna(how='all', axis=0).dropna(how='all', axis=1)
        tickers = list(denoised.columns)
        dist_df = np.sqrt(2 * (1 - denoised.clip(-1, 1)))
        return dist_df, tickers, "hierarchical", "ward", {"lam_plus": lam_plus, "eigvals": eigvals}

    elif method == "dtw_kmeans":
        if not raw_close_data:
            raise ValueError("Brak surowych cen — uruchom ponownie Stage 1.")
        prices_full = pd.DataFrame(raw_close_data).set_index('Date')
        prices_full.index = pd.to_datetime(prices_full.index)
        window = prices_full.tail(504).dropna(axis=1, how='any')
        tickers = list(window.columns)
        if len(tickers) < 2:
            raise ValueError("Za mało kompletnych serii cenowych do DTW (min. 2 spółki bez braków w 2Y).")
        cum_returns = (window / window.iloc[0]) - 1.0
        z_norm = (cum_returns - cum_returns.mean()) / cum_returns.std(ddof=0)
        dist_df = compute_dtw_distance_matrix(z_norm, window=20)
        return dist_df, tickers, "partition", None, {"z_norm": z_norm, "window": window}

    elif method == "dbscan":
        raise ValueError("DBSCAN dobiera liczbę klastrów automatycznie (eps auto-detect) — sugestia k nie ma zastosowania.")

    else:
        raise ValueError(f"Nieznana metoda: {method}")

def suggest_optimal_k(dist_df, kind, linkage_method, k_range=range(2, 7)):
    """Silhouette score (metric='precomputed') dla każdego k w zakresie, zwraca (best_k, scores_dict)."""
    from sklearn.metrics import silhouette_score
    D = dist_df.values
    n = D.shape[0]
    scores = {}
    for k in k_range:
        if k >= n:
            continue
        try:
            if kind == "hierarchical":
                Z = linkage(squareform(D, checks=False), method=linkage_method)
                labels = fcluster(Z, t=k, criterion='maxclust')
            else:  # partition (DTW + K-Medoids)
                labels, _ = kmedoids(D, k)
            if len(set(labels)) < 2:
                continue
            scores[k] = float(silhouette_score(D, labels, metric='precomputed'))
        except Exception:
            continue
    if not scores:
        return None, {}
    best_k = max(scores, key=scores.get)
    return best_k, scores



@app.callback(
    Output("data-inspector-sidebar", "className"),
    Input("btn-inspector-open", "n_clicks"),
    Input("btn-inspector-close", "n_clicks"),
    State("data-inspector-sidebar", "className"),
    prevent_initial_call=True
)
def toggle_inspector_sidebar(open_clicks, close_clicks, current_class):
    ctx = dash.callback_context
    if not ctx.triggered: return current_class
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    return "data-inspector-panel open" if trigger_id == "btn-inspector-open" else "data-inspector-panel"


@app.callback(
    Output("checklist-universe-tickers", "options"),
    Output("stage1-universe-count", "children"),
    Input("stage1-universe-search", "value"),
    Input("store-stage1-universe-refresh", "data"),
    Input("checklist-universe-tickers", "value"),
    prevent_initial_call=False,
)
def render_universe_checklist(search_value, _refresh, checked_values):
    """
    Populuje TYLKO `options` listy uniwersum -- nigdy nie dotyka `value`
    (aktualnie zaznaczonych tickerow), zeby filtrowanie wyszukiwarka nie
    czyscilo zaznaczenia zrobionego wczesniej. Checked_values jest Inputem
    tylko po to, zeby licznik "X zaznaczonych" byl zywy -- sam callback nic
    do niego nie zapisuje.
    """
    companies = uni.list_companies()
    if search_value:
        needle = search_value.strip().upper()
        companies = [c for c in companies if needle in c.get("Ticker", "").upper()]
    options = [{"label": f"{c['Ticker']}" + (f" — {c['Name']}" if c.get("Name") else ""), "value": c["Ticker"]} for c in companies]
    count_label = f"{len(checked_values or [])} zaznaczonych / {len(uni.list_companies())} w uniwersum"
    return options, count_label


@app.callback(
    Output("checklist-universe-tickers", "value"),
    Output("store-stage1-universe-refresh", "data"),
    Output("stage1-new-ticker-status", "children"),
    Input("stage1-new-ticker-btn", "n_clicks"),
    State("stage1-new-ticker-input", "value"),
    State("checklist-universe-tickers", "value"),
    State("store-stage1-universe-refresh", "data"),
    prevent_initial_call=True,
)
def add_new_universe_ticker(_n_clicks, new_ticker_value, currently_checked, refresh_counter):
    """
    Dodaje nowy ticker do uniwersum (auto-fetch Name/Sector przez yfinance,
    ten sam mechanizm co "+ SLEDZ" w Research) i od razu go zaznacza na
    liscie -- skoro dodajesz go w tym momencie, najpewniej chcesz go od razu
    uwzglednic w biezacym rebalansie, nie tylko dopisac do bazy.

    Walidacja przez parse_single_ticker_input (ui/components.py) -- poprawka
    po realnym bledzie znalezionym w testach: wpisanie "AVGO, CRDO" bylo
    wczesniej cicho akceptowane jako JEDEN literalny ticker "AVGO, CRDO"
    (yfinance dopasowywal go luzno do Broadcom przy auto-fetch nazwy, ale w
    bazie zostawal zapisany ticker z przecinkiem i drugim symbolem w srodku).
    """
    ticker, error = parse_single_ticker_input(new_ticker_value)
    if error:
        return dash.no_update, dash.no_update, html.Div(error, style={"color": THEME["neg"]})

    profile = fetch_company_profile(ticker)
    uni.upsert_company(ticker, name=profile["name"], sector=profile["sector"], status="Watchlist")

    updated_checked = list(currently_checked or [])
    if ticker not in updated_checked:
        updated_checked.append(ticker)
    return updated_checked, (refresh_counter or 0) + 1, html.Div(f"Dodano {ticker} do uniwersum.", style={"color": THEME["pos"]})


@app.callback(
    Output("dropdown-active-asset", "options"), Output("dropdown-active-asset", "value"), Output("validated-tags-container", "children"),
    Output("panel-preview-container", "style"), Output("panel-matrix-container", "style"), Output("panel-config-stage2-container", "style"),
    Output("matrix-date-range-sub", "children"), Output("table-correlation-wrapper", "children"),
    Output("store-raw-close", "data"), Output("store-monthly-returns", "data"),
    Output("store-daily-returns", "data"), Output("store-semicov-matrix", "data"),
    Output("error-output", "children"),
    Input("btn-validate", "n_clicks"), State("checklist-universe-tickers", "value")
)
def run_stage_01_ingestion(n_clicks, tickers):
    if n_clicks == 0: return dash.no_update, dash.no_update, dash.no_update, {"display": "none"}, {"display": "none"}, {"display": "none"}, "", "", None, None, None, None, ""
    if not tickers: return [], None, [], {"display": "none"}, {"display": "none"}, {"display": "none"}, "", "", None, None, None, None, "Zaznacz przynajmniej jedną spółkę z uniwersum."
    tickers = [t.strip().upper() for t in tickers if t.strip()]

    if not tickers: return [], None, [], {"display": "none"}, {"display": "none"}, {"display": "none"}, "", "", None, None, None, None, "No tickers found."

    try:
        # OPTYMALIZACJA WYKREŚLENIA INTERFEJSU: Pobieramy dane bez wielowątkowości, by OS nie ubijał procesu Pythona
        df_raw = yf.download(tickers, period="5y", group_by="ticker", threads=False, auto_adjust=True)

        if df_raw is None or df_raw.empty:
            print("[STAGE 1] yfinance zwrócił pusty DataFrame — sprawdź połączenie / czy Yahoo nie blokuje requestów.")
            return [], None, [], {"display": "none"}, {"display": "none"}, {"display": "none"}, "", "", None, None, None, None, "SYSTEM ERR // BRAK DANYCH Z YFINANCE (sprawdź konsolę)."

        # Sprawdzamy, które tickers faktycznie pobrały dane
        # UWAGA: yfinance z group_by="ticker" domyślnie ZAWSZE zwraca MultiIndex kolumn
        # (multi_level_index=True jest domyślne), nawet dla listy z JEDNYM tickerem.
        # Dlatego sprawdzamy realną strukturę kolumn, a nie len(tickers) — inaczej dla
        # jednej spółki próba odczytu płaskiego df_raw['Close'] cicho się wywala.
        is_multi_col = isinstance(df_raw.columns, pd.MultiIndex)
        if is_multi_col:
            available_top = set(df_raw.columns.get_level_values(0))
            valid_tickers = [t for t in tickers if t in available_top and not df_raw[t].dropna(how='all').empty]
        else:
            valid_tickers = tickers if not df_raw.empty else []

        tags = [html.Div(vt, style={"padding": "10px 20px", "background": "rgba(111, 44, 255, 0.12)", "border": f"1px solid {THEME['accent']}", "borderRadius": "24px", "fontSize": "13px"}) for vt in valid_tickers]
        display = {"display": "block", "marginBottom": "35px"}

        if not valid_tickers: return [], None, [], {"display": "none"}, {"display": "none"}, {"display": "none"}, "", "", None, None, None, None, "SYSTEM ERR // NO VALID TICKERS FOUND."

        # Budujemy czysty DataFrame cen zamknięcia z pobranej paczki danych
        prices = pd.DataFrame(index=df_raw.index)
        for t in valid_tickers:
            if is_multi_col:
                prices[t] = df_raw[t]['Close']
            else:
                prices[t] = df_raw['Close']

        prices = prices.dropna(how='all').ffill().bfill()

        prices_df_clean = prices.reset_index()
        prices_df_clean['Date'] = prices_df_clean['Date'].dt.strftime('%Y-%m-%d')
        prices_json = prices_df_clean.to_dict('records')

        date_sub = f"Based on monthly returns from {prices.index.min().strftime('%b %Y')} to {prices.index.max().strftime('%b %Y')}"

        # --- ZWROTY MIESIĘCZNE (bez zmian — zasila tabelę korelacji Stage 1) ---
        monthly_prices = prices.resample('ME').last()
        monthly_returns = np.log(monthly_prices / monthly_prices.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()

        returns_df_clean = monthly_returns.reset_index()
        returns_df_clean['Date'] = returns_df_clean['Date'].dt.strftime('%Y-%m-%d')
        returns_json = returns_df_clean.to_dict('records')

        # --- ZWROTY DZIENNE (nowe — zasilają semi-covariance i przyszłe metody: RMT, DBSCAN, DTW) ---
        daily_returns = np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()

        daily_returns_clean = daily_returns.reset_index()
        daily_returns_clean['Date'] = daily_returns_clean['Date'].dt.strftime('%Y-%m-%d')
        daily_returns_json = daily_returns_clean.to_dict('records')

        # --- MACIERZ SEMI-KOWARIANCJI (Estrada, downside risk na dziennych zwrotach) ---
        semicov_matrix = compute_semicovariance_matrix(daily_returns)
        semicov_df_clean = semicov_matrix.reset_index().rename(columns={'index': 'Ticker'})
        semicov_json = semicov_df_clean.to_dict('records')

        corr_matrix = monthly_returns.corr().dropna(how='all', axis=0).dropna(how='all', axis=1)
        active_ts = list(corr_matrix.columns)

        grid_df = corr_matrix.reset_index().rename(columns={'index': 'Ticker'})
        columns = [{"name": "Ticker", "id": "Ticker"}] + [{"name": t, "id": t, "type": "numeric", "format": {"specifier": ".2f"}} for t in active_ts]

        data_records = [{'Ticker': r['Ticker'], **{t: r[t] for t in active_ts}} for _, r in grid_df.iterrows()]
        table = dash_table.DataTable(
            columns=columns, data=data_records, style_data_conditional=[datatable_row_alt_rule()] + generate_tws_matrix_styles(grid_df, active_ts),
            style_header=datatable_style_header(),
            style_data=datatable_style_data(),
            style_cell={'textAlign': 'center', 'fontVariantNumeric': 'tabular-nums'},
            style_cell_conditional=[{'if': {'column_id': 'Ticker'}, 'textAlign': 'left', 'fontWeight': 'bold', 'color': THEME['text_dim'], 'width': '110px'}]
        )
        return ([{"label": tx, "value": tx} for tx in valid_tickers], valid_tickers[0], tags, display, display, display,
                date_sub, table, prices_json, returns_json, daily_returns_json, semicov_json, "")
    except Exception as e:
        print("=" * 60)
        print("[STAGE 1] CRITICAL ERROR - PEŁNY TRACEBACK:")
        traceback.print_exc()
        print("=" * 60)
        return [], None, [], {"display": "none"}, {"display": "none"}, {"display": "none"}, "", "", None, None, None, None, f"STAGE 1 CRITICAL ERR // {str(e)}"


@app.callback(
    Output("dropdown-k-count", "options"), Output("dropdown-k-count", "value"), Output("label-k-count", "children"),
    Input("dropdown-cluster-method", "value"), State("dropdown-k-count", "value")
)
def adjust_k_count_for_method(method, current_k):
    base_options = [{"label": str(i), "value": i} for i in range(2, 7)]
    if method == "dbscan":
        options = [{"label": "Auto (data-driven, k-distance elbow)", "value": "auto"}] + base_options
        return options, "auto", "Cluster Count — Auto-detected (opcjonalnie wymuś liczbę):"
    if current_k == "auto" or current_k is None:
        return base_options, 3, "Target Cluster Count (k):"
    return base_options, current_k, "Target Cluster Count (k):"


@app.callback(
    Output("suggested-k-output", "children"),
    Input("btn-suggest-k", "n_clicks"),
    State("dropdown-cluster-method", "value"), State("store-monthly-returns", "data"),
    State("store-daily-returns", "data"), State("store-raw-close", "data"),
    prevent_initial_call=True
)
def suggest_k_callback(n_clicks, method, monthly_returns_data, daily_returns_data, raw_close_data):
    if not SKLEARN_AVAILABLE:
        return html.Span("scikit-learn niedostępny — silhouette score wymaga: pip install scikit-learn", style={"color": THEME["orange"]})
    if method == "dbscan":
        return html.Span("DBSCAN dobiera liczbę klastrów automatycznie (patrz k-distance elbow) — sugestia k się tu nie stosuje.", style={"color": THEME["text_dim"]})
    try:
        dist_df, tickers, kind, linkage_method, extra = get_method_distance_matrix(method, monthly_returns_data, daily_returns_data, raw_close_data)
    except ValueError as e:
        return html.Span(f"Nie można policzyć: {str(e)}", style={"color": THEME["orange"]})

    best_k, scores = suggest_optimal_k(dist_df, kind, linkage_method)
    if best_k is None:
        return html.Span("Za mało spółek / za mało wariancji, żeby policzyć silhouette score.", style={"color": THEME["orange"]})

    scores_txt = "  |  ".join([f"k={k}: {v:.3f}" + (" (best)" if k == best_k else "") for k, v in sorted(scores.items())])
    return html.Div([
        html.Span(f"Sugerowane k = {best_k} (silhouette score = {scores[best_k]:.3f}). ", style={"color": THEME["accent"], "fontWeight": "bold"}),
        html.Div(scores_txt, style={"marginTop": "4px", "fontSize": "11px"})
    ])


@app.callback(
    Output("panel-clustering-outputs-container", "style"), Output("graph-dendrogram", "figure"), Output("graph-clustered-heatmap", "figure"),
    Output("cluster-legend-output", "children"), Output("store-dist-matrix", "data"), Output("error-output", "children", allow_duplicate=True),
    Input("btn-cluster-execute", "n_clicks"), State("dropdown-cluster-method", "value"), State("dropdown-k-count", "value"),
    State("store-monthly-returns", "data"), State("store-daily-returns", "data"), State("store-raw-close", "data"),
    prevent_initial_call=True
)
def run_stage_02_clustering(n_clicks, method, k_count, monthly_returns_data, daily_returns_data, raw_close_data):
    if n_clicks == 0: return {"display": "none"}, go.Figure(), go.Figure(), "", None, ""

    HIERARCHICAL_LINKAGE = {"ward": "ward", "complete": "complete", "single": "single", "average": "average"}
    HIERARCHICAL_LABEL = {"ward": "Ward Linkage", "complete": "Complete Linkage", "single": "Single Linkage", "average": "Average Linkage"}

    try:
        dendro_fig = None  # ustawiane w gałęzi bez naturalnego dendrogramu (DBSCAN, DTW)
        k_safe = int(k_count) if isinstance(k_count, (int, float)) else 3

        # ============ 4 METODY BAZOWE, SEMI-COV, RMT — wszystkie hierarchiczne (Ward/Complete/Single/Average) ============
        if method in HIERARCHICAL_LINKAGE or method in ("semicov", "rmt_denoised"):
            try:
                dist_matrix, active_tickers, kind, linkage_method, extra = get_method_distance_matrix(method, monthly_returns_data, daily_returns_data, raw_close_data)
            except ValueError as e:
                return {"display": "none"}, go.Figure(), go.Figure(), "", None, f"STAGE 2 ERR // {str(e)}"

            display_matrix = 1 - (dist_matrix.values ** 2) / 2  # odwrotność D=sqrt(2(1-rho)) -> rho, dla heatmapy
            display_matrix = pd.DataFrame(display_matrix, index=active_tickers, columns=active_tickers)
            Z = linkage(squareform(dist_matrix.values, checks=False), method=linkage_method)
            labels = fcluster(Z, t=k_safe, criterion='maxclust')
            ticker_to_cluster = {active_tickers[i]: int(labels[i]) for i in range(len(active_tickers))}
            ordered_tickers = [active_tickers[i] for i in leaves_list(Z)]
            zmin, zmax = -1.0, 1.0
            dendro_fig = build_dendrogram_figure(Z, active_tickers, ticker_to_cluster)

            if method in HIERARCHICAL_LINKAGE:
                model_label = f"{HIERARCHICAL_LABEL[method]} — Standard Correlation (Monthly)"
            elif method == "semicov":
                model_label = "Semi-Covariance / Downside Risk (Daily, Ward Linkage)"
            else:
                n_noise_eig = int((extra["eigvals"] <= extra["lam_plus"]).sum())
                model_label = f"RMT Denoised (Marchenko-Pastur, Ward) — {n_noise_eig}/{len(extra['eigvals'])} wartości własnych odfiltrowano jako szum"

        # ============ DBSCAN — Density-Based (Monthly), auto lub wymuszone k ============
        elif method == "dbscan":
            if not SKLEARN_AVAILABLE:
                return {"display": "none"}, go.Figure(), go.Figure(), "", None, "STAGE 2 ERR // scikit-learn nie jest zainstalowany. Zrób: pip install scikit-learn"
            if not monthly_returns_data:
                return {"display": "none"}, go.Figure(), go.Figure(), "", None, "STAGE 2 ERR // Brak miesięcznych zwrotów — uruchom ponownie Stage 1."
            df_returns = pd.DataFrame(monthly_returns_data).set_index('Date')
            display_matrix = df_returns.corr().dropna(how='all', axis=0).dropna(how='all', axis=1)
            active_tickers = list(display_matrix.columns)
            dist_matrix = np.sqrt(2 * (1 - display_matrix.clip(-1, 1)))
            D = dist_matrix.values
            D_masked = D.copy()
            np.fill_diagonal(D_masked, np.inf)
            nn_dist = np.sort(np.min(D_masked, axis=1))
            eps_auto = compute_elbow_eps(nn_dist)

            forced = isinstance(k_count, (int, float))
            if forced:
                Z_fb = linkage(squareform(D, checks=False), method='ward')
                labels0 = fcluster(Z_fb, t=int(k_count), criterion='maxclust') - 1
            else:
                labels0 = DBSCAN(eps=float(eps_auto), min_samples=2, metric='precomputed').fit_predict(D)

            ticker_to_cluster = {active_tickers[i]: int(labels0[i]) for i in range(len(active_tickers))}
            n_found = len(set(l for l in labels0 if l != -1))
            n_noise = int((labels0 == -1).sum())
            if forced:
                model_label = f"DBSCAN [WYMUSZONE {int(k_count)} klastrów → fallback Ward] — Standard Correlation (Monthly)"
            else:
                model_label = f"DBSCAN (auto, eps={eps_auto:.3f}, MinSamples=2) — wykryto {n_found} klastrów, {n_noise} spółek jako szum"
            ordered_tickers = sorted(active_tickers, key=lambda t: (ticker_to_cluster[t] == -1, ticker_to_cluster[t]))
            zmin, zmax = -1.0, 1.0

            dendro_fig = go.Figure()
            if not forced:
                dendro_fig.add_trace(go.Scatter(x=list(range(len(nn_dist))), y=nn_dist, mode='lines+markers', line=dict(color=THEME["accent"], width=2), marker=dict(size=5), name="k-distance"))
                dendro_fig.add_hline(y=eps_auto, line_dash="dash", line_color=THEME["orange"], annotation_text=f"auto eps = {eps_auto:.3f}", annotation_font_color=THEME["orange"])
                dendro_fig.update_layout(title=dict(text="K-Distance Elbow Plot (auto-dobór eps)", font=dict(color=THEME["text_white"], size=13)))
            else:
                dendro_fig.add_annotation(text="Wymuszona liczba klastrów → fallback na Ward linkage.<br>Brak naturalnego k-distance plot w tym trybie.", showarrow=False, font=dict(color=THEME["text_dim"], size=13))
            dendro_fig.update_layout(template="plotly_dark", height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"], font_family=THEME["font"], margin=dict(l=50, r=30, t=40, b=40))

        # ============ TIME-SERIES DTW + K-MEDOIDS (Daily Price Shape) ============
        elif method == "dtw_kmeans":
            try:
                dtw_dist_df, active_tickers, kind, _, extra = get_method_distance_matrix(method, monthly_returns_data, daily_returns_data, raw_close_data)
            except ValueError as e:
                return {"display": "none"}, go.Figure(), go.Figure(), "", None, f"STAGE 2 ERR // {str(e)}"
            z_norm, window = extra["z_norm"], extra["window"]

            D = dtw_dist_df.values
            k_eff = int(k_count) if isinstance(k_count, (int, float)) else 3
            labels0, medoid_idx = kmedoids(D, k_eff)
            ticker_to_cluster = {active_tickers[i]: int(labels0[i]) for i in range(len(active_tickers))}

            max_d = D.max() if D.max() > 0 else 1.0
            display_matrix = pd.DataFrame(1 - (D / max_d), index=active_tickers, columns=active_tickers)
            model_label = f"Time-Series DTW + K-Medoids (k={k_eff}) — Daily Price Shape, ostatnie {len(window)} sesji"
            ordered_tickers = sorted(active_tickers, key=lambda t: ticker_to_cluster[t])
            zmin, zmax = 0.0, 1.0
            dist_matrix = dtw_dist_df

            dendro_fig = go.Figure()
            for i, t in enumerate(active_tickers):
                c = int(labels0[i])
                color = CLUSTER_PALETTE.get(c + 1, "#888888")
                dendro_fig.add_trace(go.Scatter(x=list(range(len(z_norm))), y=z_norm[t].values, mode='lines', line=dict(color=color, width=1.5), name=t, opacity=0.85))
            dendro_fig.update_layout(
                template="plotly_dark", height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"], font_family=THEME["font"],
                margin=dict(l=50, r=30, t=40, b=40), showlegend=True, legend=dict(font=dict(color=THEME["text_white"], size=10)),
                title=dict(text="Znormalizowane trajektorie cenowe (kolor = klaster DTW)", font=dict(color=THEME["text_white"], size=13)),
                xaxis=dict(showgrid=False, tickfont=dict(color=THEME["text_dim"])), yaxis=dict(gridcolor="#1E1E28", tickfont=dict(color=THEME["text_dim"]))
            )
        else:
            return {"display": "none"}, go.Figure(), go.Figure(), "", None, f"STAGE 2 ERR // Nieznana metoda: {method}"

        # ============ WSPÓLNE: legenda, heatmapa, zapis macierzy odległości ============
        legend_blocks = build_cluster_legend(ticker_to_cluster, model_label)

        reordered_matrix = display_matrix.loc[ordered_tickers, ordered_tickers]
        # Skala diverging (-1..1, korelacje) vs sequential (0..1, dystans/DBSCAN) --
        # patrz ui/theme.py: MATRIX_COLORSCALE zaklada neutralne 0 w POLOWIE zakresu
        # (poprawne tylko dla -1..1), MATRIX_SEQUENTIAL_COLORSCALE zaklada neutralne 0
        # na SAMYM POCZATKU zakresu (poprawne dla 0..max, gdzie nie ma ujemnej strony).
        heatmap_colorscale = MATRIX_COLORSCALE if zmin < 0 else MATRIX_SEQUENTIAL_COLORSCALE
        fig_clustered_map = go.Figure(data=go.Heatmap(
            z=reordered_matrix.values, x=ordered_tickers, y=ordered_tickers, zmin=zmin, zmax=zmax,
            colorscale=heatmap_colorscale, showscale=False, hoverongaps=False,
            text=reordered_matrix.values, texttemplate="%{text:.2f}", textfont=dict(size=10, color=THEME["text_white"]),
            xgap=2, ygap=2,
        ))
        fig_clustered_map.update_layout(
            template="plotly_dark", width=450, height=450, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"],
            margin=dict(l=40, r=40, t=40, b=40), font_family=THEME["font"],
            xaxis=dict(showgrid=False, tickfont=dict(color=THEME["text_white"], size=11)),
            yaxis=dict(showgrid=False, tickfont=dict(color=THEME["text_white"], size=11), autorange="reversed")
        )

        dist_out_df = dist_matrix if method == "dtw_kmeans" else pd.DataFrame(dist_matrix, index=active_tickers, columns=active_tickers)
        dist_matrix_df = dist_out_df.reset_index().rename(columns={'index': 'Ticker'})
        dist_json = dist_matrix_df.to_dict('records')

        return {"display": "block"}, dendro_fig, fig_clustered_map, legend_blocks, dist_json, ""
    except Exception as e:
        print("=" * 60)
        print("[STAGE 2] CRITICAL ERROR - PEŁNY TRACEBACK:")
        traceback.print_exc()
        print("=" * 60)
        return {"display": "none"}, go.Figure(), go.Figure(), "", None, f"STAGE 2 CRITICAL ERR // {str(e)}"


@app.callback(
    Output("inspector-table-container", "children"),
    Input("dropdown-inspector-dataset", "value"),
    Input("store-raw-close", "data"),
    Input("store-monthly-returns", "data"),
    Input("store-daily-returns", "data"),
    Input("store-semicov-matrix", "data"),
    Input("store-dist-matrix", "data")
)
def render_inspector_raw_data(selected_type, raw_close, monthly_returns, daily_returns, semicov_matrix, dist_matrix):
    target_data = None
    if selected_type == "raw_close": target_data = raw_close
    elif selected_type == "monthly_returns": target_data = monthly_returns
    elif selected_type == "daily_returns": target_data = daily_returns
    elif selected_type == "semicov_matrix": target_data = semicov_matrix
    elif selected_type == "dist_matrix": target_data = dist_matrix
    
    if not target_data:
        return html.Div("No data available. Run [INITIALIZE DATASETS] first.", style={"color": THEME["orange"], "padding": "20px", "textAlign": "center"})

    # Precyzja dobrana per typ danych — semi-covariance to bardzo małe liczby (rzędu wariancji
    # dziennych zwrotów, ~1e-4), przy 4 miejscach po przecinku różnice między spółkami się spłaszczają.
    FORMAT_SPECIFIERS = {
        "raw_close": ".2f", "monthly_returns": ".6f", "daily_returns": ".6f",
        "semicov_matrix": ".8f", "dist_matrix": ".6f"
    }
    specifier = FORMAT_SPECIFIERS.get(selected_type, ".6f")

    df = pd.DataFrame(target_data)
    columns = []
    for c in df.columns:
        if c in ['Date', 'Ticker']: columns.append({"name": c, "id": c})
        else: columns.append({"name": c, "id": c, "type": "numeric", "format": {"specifier": specifier}})
            
    return dash_table.DataTable(
        columns=columns, data=df.to_dict('records'), page_action='native', page_size=15,
        style_header=datatable_style_header(),
        style_data=datatable_style_data(),
        style_cell=datatable_style_cell(),
        style_data_conditional=[datatable_row_alt_rule()]
    )


@app.callback(Output("dropdown-active-asset", "multi"), Output("dropdown-active-asset", "value", allow_duplicate=True), Input("checkbox-overlay-mode", "value"), State("dropdown-active-asset", "options"), State("dropdown-active-asset", "value"), prevent_initial_call=True)
def handle_overlay_mode(overlay_value, options, current_val):
    is_overlay = "OVERLAY" in overlay_value
    if not options: return is_overlay, dash.no_update
    first_ticker = options[0]['value']
    next_val = [current_val] if isinstance(current_val, str) else ([first_ticker] if not current_val else current_val) if is_overlay else current_val[0] if isinstance(current_val, list) and current_val else first_ticker
    return is_overlay, next_val

TIMEFRAME_SESSIONS = {"5Y": 1260, "2Y": 504, "1Y": 252, "6M": 126, "3M": 63, "1M": 21}


@app.callback(
    Output("graph-asset-preview", "figure"),
    Input("dropdown-active-asset", "value"), Input("radio-chart-type", "value"), Input("dropdown-chart-timeframe", "value"),
    Input("store-raw-close", "data"),
    prevent_initial_call=True
)
def render_multi_asset_chart(active_assets, chart_scale, timeframe, raw_close_data):
    if not active_assets or not raw_close_data: return go.Figure()
    asset_list = active_assets if isinstance(active_assets, list) else [active_assets]

    prices_full = pd.DataFrame(raw_close_data).set_index('Date')
    prices_full.index = pd.to_datetime(prices_full.index)
    n_sessions = TIMEFRAME_SESSIONS.get(timeframe, 1260)
    window = prices_full.tail(n_sessions)

    fig = go.Figure()
    for idx, ticker in enumerate(asset_list):
        try:
            if ticker not in window.columns: continue
            y_data = window[ticker].dropna()
            if chart_scale == "pct":
                y_plot = (y_data / y_data.iloc[0] - 1.0) * 100.0
            else:
                y_plot = y_data
            fig.add_trace(go.Scatter(
                x=y_plot.index, y=y_plot.values, mode='lines',
                line=dict(color=CHART_COLORS[idx % len(CHART_COLORS)], width=2.5),
                name=ticker
            ))
        except Exception:
            print(f"[CHART] Nie udało się narysować {ticker}:")
            traceback.print_exc()
            continue

    y_axis_type = "log" if chart_scale == "log" else "linear"
    y_axis_ticksuffix = "%" if chart_scale == "pct" else ""
    fig.update_layout(
        template="plotly_dark", height=550, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"], font_family=THEME["font"],
        showlegend=True, legend=dict(font=dict(color=THEME["text_white"], size=12)),
        yaxis=dict(type=y_axis_type, ticksuffix=y_axis_ticksuffix, showgrid=True, gridcolor="#1E1E28", side="right"),
        xaxis=dict(showgrid=True, gridcolor="#1E1E28")
    )
    return fig