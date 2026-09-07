"""
ui/tab5_sandbox.py
====================
Tab 5: Forward Tracker / Sandbox -- loads a saved portfolio snapshot,
re-fetches live prices, and lets the user re-run the exact same solver
(run_optimization_with_singleton_split) with live-adjustable parameters
without touching Stages 1-4.

Moved out of quant_terminal.py (Etap 0 architecture split, PROJECT_CONTEXT.md)
with NO behavior change.
"""
import dash
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State

from ui.app_instance import app
from ui.theme import THEME
from ui.components import build_kpi_card, build_kpi_strip, datatable_style_header, datatable_style_cell, datatable_style_data, datatable_row_alt_rule
from engine.risk import compute_estrada_matrix, compute_crash_overlap_matrix
from engine.optimizer import run_optimization_with_singleton_split
from engine.returns import compute_composite_upside_row
from data import snapshot_store as snap

# NOTE (2026-09-07, Etap 3 follow-up): the panel this module renders into
# (panel-stage4b-tracker-container) used to be gated behind Stage 3
# confirmation via a `reveal_stage4b_tracker_panel` callback -- that made
# sense when Sandbox was tab-5 in a single shared sequential tab bar with
# Tabs 1-4. Now that Sandbox is its own independent sidebar module
# (ui/layout.py's module-sandbox), a user can open it directly to load ANY
# previously saved snapshot from disk without ever touching the Rebalance
# workflow in the current session -- gating it behind "did you just confirm
# Stage 3" would show an empty panel for exactly that intended use case.
# The panel is now always visible by default (ui/layout.py); this callback
# was removed rather than kept-but-unused, since a dead Output/Input wiring
# left behind is worse than no trace of it -- the git history is the record
# of why, not a commented-out function.


@app.callback(
    Output("dropdown-snapshot-select", "options"), Output("dropdown-snapshot-select", "value"),
    Input("store-snapshots-refresh", "data"),
)
def load_snapshot_list(_):
    snapshots = snap.list_snapshots()
    options = [{"label": f"{s['snapshot_name']}  ({s['snapshot_id']})", "value": s["snapshot_id"]} for s in snapshots]
    value = options[0]["value"] if options else None
    return options, value


@app.callback(
    Output("store-snapshots-refresh", "data", allow_duplicate=True),
    Input("btn-rename-snapshot", "n_clicks"),
    State("dropdown-snapshot-select", "value"), State("input-rename-snapshot", "value"), State("store-snapshots-refresh", "data"),
    prevent_initial_call=True
)
def rename_snapshot_callback(n_clicks, snapshot_id, new_name, counter):
    if not snapshot_id or not new_name or not new_name.strip():
        return dash.no_update
    snap.rename_snapshot(snapshot_id, new_name)
    return (counter or 0) + 1


@app.callback(
    Output("store-delete-armed", "data"), Output("delete-confirm-banner", "children"),
    Output("store-snapshots-refresh", "data", allow_duplicate=True),
    Input("btn-delete-snapshot", "n_clicks"), Input("dropdown-snapshot-select", "value"),
    State("store-delete-armed", "data"), State("store-snapshots-refresh", "data"),
    prevent_initial_call=True
)
def delete_snapshot_guarded(n_clicks_delete, selected_id, armed_id, counter):
    """
    Usuwanie z dwustopniowym zabezpieczeniem: pierwsze kliknięcie DELETE tylko "uzbraja"
    usunięcie tego konkretnego snapshotu i pokazuje pasek potwierdzenia; dopiero DRUGIE
    kliknięcie na już-uzbrojony snapshot faktycznie usuwa. Zmiana wyboru w dropdownie
    (np. przypadkowe kliknięcie gdzie indziej) automatycznie rozbraja -- nic nie usunie
    się przez pomyłkę przy zwykłym przeglądaniu listy.
    """
    trigger_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0] if dash.callback_context.triggered else None

    if trigger_id == "dropdown-snapshot-select":
        return None, "", dash.no_update

    if not selected_id:
        return None, "", dash.no_update

    if armed_id != selected_id:
        # Pierwsze kliknięcie -> uzbrojenie, BRAK usunięcia
        banner = html.Div(style={"padding": "10px", "backgroundColor": "rgba(255,138,0,0.12)", "border": f"1px solid {THEME['orange']}", "borderRadius": "10px"}, children=[
            html.Span("Kliknij DELETE jeszcze raz, żeby na pewno usunąć ten snapshot. ", style={"color": THEME["orange"], "fontSize": "11px", "fontWeight": "bold"}),
            html.Button("ANULUJ", id="btn-cancel-delete", n_clicks=0, style={
                "marginLeft": "8px", "padding": "4px 10px", "backgroundColor": "transparent", "color": THEME["text_dim"],
                "border": f"1px solid {THEME['border']}", "borderRadius": "6px", "cursor": "pointer", "fontSize": "10px"
            })
        ])
        return selected_id, banner, dash.no_update

    # Drugie kliknięcie na już-uzbrojony snapshot -> faktyczne usunięcie
    snap.delete_snapshot(selected_id)
    return None, html.Div("Usunięto.", style={"color": THEME["text_dim"], "fontSize": "11px"}), (counter or 0) + 1


@app.callback(
    Output("store-delete-armed", "data", allow_duplicate=True), Output("delete-confirm-banner", "children", allow_duplicate=True),
    Input("btn-cancel-delete", "n_clicks"), prevent_initial_call=True
)
def cancel_delete_callback(n_clicks):
    return None, ""


def cap_weights_iteratively(weights, cap, max_iter=100):
    """
    Wymusza w_i <= cap na WSZYSTKICH wagach, zachowując sum(w)=1 -- prosty, naiwny
    'clip potem renormalizuj w jednym kroku' TEGO NIE GWARANTUJE (renormalizacja po
    przycięciu może z powrotem wypchnąć przyciętą wagę ponad cap, jeśli koncentracja
    jest wysoka). Standardowy iteracyjny 'water-filling': elementy, które przekroczą cap,
    są TRWALE BLOKOWANE dokładnie na wartości cap (nigdy więcej nie dostają nadwyżki do
    redystrybucji) -- bez trwałego blokowania nadwyżka potrafi oscylować między dwoma
    dużymi wagami w nieskończoność zamiast zbiec (to właśnie się działo w pierwszej,
    naiwnej wersji tej funkcji). Jeśli N*cap < 1 (matematycznie niewykonalne, żeby
    zsumować do 1 przy tym capie), zwraca equal-weight jako bezpieczny fallback.
    Zwraca (weights, was_infeasible: bool) -- caller powinien poinformować użytkownika
    gdy was_infeasible=True, zamiast cicho podstawiać equal-weight bez wyjaśnienia.
    """
    n = len(weights)
    if n == 0:
        return weights, False
    if n * cap < 1.0 - 1e-9:
        return pd.Series(1.0 / n, index=weights.index), True

    w = weights.copy().astype(float)
    total = w.sum()
    w = w / total if total > 1e-9 else pd.Series(1.0 / n, index=weights.index)

    locked = pd.Series(False, index=w.index)  # elementy raz przycięte -- zamrożone na cap na zawsze

    for _ in range(max_iter):
        free_mask = ~locked
        if not free_mask.any():
            break
        over_mask = free_mask & (w > cap + 1e-12)
        if not over_mask.any():
            break
        locked = locked | over_mask
        excess = (w[over_mask] - cap).sum()
        w[over_mask] = cap
        free_mask = ~locked
        if not free_mask.any():
            break
        free_sum = w[free_mask].sum()
        if free_sum <= 1e-12:
            # Wolne elementy mają ~zerową wagę -- proporcjonalna redystrybucja (proporcja do
            # zera) nie ma jak wstrzyknąć nadwyżki, więc dzielimy ją RÓWNO między nie zamiast
            # dać jej po cichu zniknąć (co inaczej psuje sum(w)=1 i maskująca renormalizacja
            # na końcu z powrotem przepycha zablokowane wagi ponad cap).
            n_free = int(free_mask.sum())
            w[free_mask] = w[free_mask] + excess / n_free
        else:
            w[free_mask] = w[free_mask] + excess * (w[free_mask] / free_sum)

    total = w.sum()
    return (w / total) if total > 1e-9 else pd.Series(1.0 / n, index=weights.index), False


def compute_sandbox_allocation(record, tickers, risk_prices, sb_lambda, sb_gamma, sb_kappa, sb_wmax, sb_rf, sb_nref, sb_alpha):
    """
    Przelicza wagi 'sandbox' NA ŻYWO -- dokładnie tym samym silnikiem
    run_optimization_with_singleton_split() (True Two-Stage SLSQP + Section
    3.3 crash-overlap matrix K + Dynamic Singleton Split) co główny solver w
    Tab 4 (run_stage4b_solver) -- "Unifikacja Sandboxa", żadnej osobnej
    uproszczonej ścieżki. P_i (stary Z-score penalty) NIE wchodzi już do
    solvera -- to teraz czysto historyczny artefakt, nieużywany tutaj wcale
    (zastąpiony przez K; zostawiony tylko w Tab 4's diagnostycznej tabeli).

    `sb_rf` / `sb_nref` to TERAZ żywe suwaki sandboxa (Tab 5), a NIE wartości
    zamrożone w `record["parameters"]` w dniu zapisu -- wcześniej R_f i N_ref
    było w ogóle niemożliwe zmienić w sandboxie, co było realnym brakiem
    funkcjonalności (nie dało się np. sprawdzić "co by było, gdyby stopa wolna
    od ryzyka wzrosła o 2pp" bez tworzenia nowego snapshotu). λ, γ, κ, w_max
    już wcześniej były suwakami -- teraz komplet 6 parametrów solvera jest
    edytowalny w locie.

    WAŻNE: `risk_prices` to SUROWE ceny (mogą zawierać NaN -- różne kalendarze
    giełd, patrz `compute_crash_overlap_matrix`), okno 5 lat wstecz (~1260 sesji)
    od dziś, używane WYŁĄCZNIE do estymacji Sigma_eps i K -- celowo NIE to samo
    okno co equity curve (które zaczyna się dopiero w dniu zapisu snapshotu).

    Zwraca (weights: pd.Series, note: str|None, extra: dict) gdzie `extra` zawiera:
        "cluster_of"       : dict {ticker: cluster_label} -- struktura PO ewentualnym
                              Singleton Split w tym konkretnym przebiegu sandboxa
                              (może się różnić od oryginalnego zapisanego cluster_of,
                              bo inne suwaki = inna koncentracja = inny wynik splitu).
        "promoted_tickers" : list[str] -- puste jeśli split się nie uruchomił.
        "mu_vec"           : dict {ticker: mu_i} pod BIEŻĄCYMI suwakami (γ, κ, N_ref).
        "mu_p", "delta_p", "k_penalty_p", "tps_p" : float|None -- metryki portfela
                              sandboxa na poziomie całości (None przy fallbacku equal-weight).
    """
    fund_by_ticker = {r["Ticker"]: r for r in record.get("fundamental_inputs", [])}
    cluster_of_orig = record.get("cluster_of", {t: 1 for t in tickers})

    mu_vec = pd.Series({t: compute_composite_upside_row(fund_by_ticker.get(t, {}), sb_gamma, sb_kappa, sb_nref, alpha=sb_alpha)["mu_i"] for t in tickers})

    equal_weight_fallback = pd.Series(1.0 / len(tickers), index=tickers)
    note = None
    extra = {"cluster_of": dict(cluster_of_orig), "promoted_tickers": [], "mu_vec": mu_vec.to_dict(),
             "mu_p": None, "delta_p": None, "k_penalty_p": None, "tps_p": None}

    if risk_prices.shape[1] >= 2 and len(risk_prices) >= 20:
        returns_window = np.log(risk_prices / risk_prices.shift(1)).dropna()
        if len(returns_window) >= 10:
            usable_tickers = list(returns_window.columns)
            clusters_dict = {}
            for t in usable_tickers:
                clusters_dict.setdefault(cluster_of_orig.get(t, 1), []).append(t)
            try:
                sigma_sb = compute_estrada_matrix(returns_window)
                crash_sb = compute_crash_overlap_matrix(risk_prices[usable_tickers], quantile=0.10)
                split_result = run_optimization_with_singleton_split(
                    mu_vec.reindex(usable_tickers), sigma_sb, crash_sb["K"], clusters_dict, sb_lambda, w_max=sb_wmax, Rf=sb_rf
                )
                w_raw = split_result["final"]["weights"].reindex(tickers).fillna(0.0)
                extra["cluster_of"] = {t: ck for ck, members in split_result["final_clusters_dict"].items() for t in members}
                extra["promoted_tickers"] = split_result["promoted_tickers"]
                extra["mu_p"] = split_result["final"]["mu_p"]
                extra["delta_p"] = split_result["final"]["delta_p"]
                extra["k_penalty_p"] = split_result["final"]["k_penalty_p"]
                extra["tps_p"] = split_result["final"]["tps_p"]
                if split_result["split_triggered"]:
                    note = f"⚡ Auto-promocja do singletona ({split_result['n_passes']} przebiegi): {', '.join(split_result['promoted_tickers'])}."
            except Exception:
                w_raw = equal_weight_fallback
                note = "Błąd w solverze True Two-Stage SLSQP — użyto equal-weight."
        else:
            w_raw = equal_weight_fallback
            note = "Za mało obserwacji zwrotów do estymacji ryzyka — użyto equal-weight."
    else:
        w_raw = equal_weight_fallback
        note = "Za mało wspólnych danych cenowych na estymację ryzyka — użyto equal-weight."

    w_final, cap_infeasible = cap_weights_iteratively(w_raw, sb_wmax)
    if cap_infeasible:
        n_assets = len(tickers)
        note = (note + " " if note else "") + f"w_max={sb_wmax:.2f} zbyt restrykcyjny dla {n_assets} aktywów (N×w_max={n_assets*sb_wmax:.2f} < 1.0) — użyto equal-weight zamiast alokacji."
    return w_final, note, extra


@app.callback(
    Output("kpi-summary-row", "children"), Output("sandbox-weights-table-container", "children"),
    Output("graph-forward-equity-curves", "figure"), Output("graph-asset-returns-bar", "figure"),
    Output("graph-forward-drawdowns", "figure"), Output("snapshot-meta-info", "children"),
    Output("sandbox-mini-kpi-row", "children"),
    Input("dropdown-snapshot-select", "value"), Input("slider-sb-alpha", "value"), Input("slider-sb-lambda", "value"), Input("slider-sb-gamma", "value"),
    Input("slider-sb-kappa", "value"), Input("slider-sb-rf", "value"), Input("slider-sb-nref", "value"),
    Input("slider-sb-wmax", "value"), Input("checklist-benchmarks", "value"), Input("btn-refresh-live", "n_clicks"),
    prevent_initial_call=True
)
def update_forward_tracker(snapshot_id, sb_alpha, sb_lambda, sb_gamma, sb_kappa, sb_rf, sb_nref, sb_wmax, benchmarks, _):
    empty_fig = go.Figure()
    empty_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"])
    empty_msg_style = {"color": THEME["text_dim"], "fontSize": "12px"}

    if not snapshot_id:
        return [], html.Div("Wybierz zapisany portfel z listy powyżej.", style=empty_msg_style), empty_fig, empty_fig, empty_fig, "", []

    record = snap.get_snapshot(snapshot_id)
    if not record:
        return [], html.Div("Nie znaleziono snapshotu (mógł zostać usunięty).", style={"color": THEME["orange"]}), empty_fig, empty_fig, empty_fig, "", []

    created_at_full = record.get("created_at", "")
    created_at = created_at_full[:10] if created_at_full else "?"
    holding_days = snap.holding_days_since(record)
    meta_info = f"Utworzono: {created_at}  •  {holding_days} dni w portfolio"

    orig_weights = pd.Series(record.get("final_weights", {}))
    tickers = list(orig_weights.index)
    if not tickers:
        return [], html.Div("Snapshot nie zawiera żadnych aktywów.", style={"color": THEME["orange"]}), empty_fig, empty_fig, empty_fig, meta_info, []

    # Pobieramy DŁUGIE okno (5 lat wstecz od dnia zapisu, aż do dziś) w JEDNYM zapytaniu --
    # ryzyko (Sigma_eps + K) liczymy z ostatniej, świeżej części tego okna (zawsze wystarczająco
    # danych, niezależnie jak młody jest snapshot), a equity curve tylko z części OD dnia zapisu.
    try:
        risk_start_dt = datetime.fromisoformat(created_at) - timedelta(days=1825)
        risk_start = risk_start_dt.strftime("%Y-%m-%d")
    except ValueError:
        risk_start = created_at

    fetch_list = list(dict.fromkeys(tickers + ["SPY", "QQQ"]))
    full_prices = snap.fetch_price_history(fetch_list, risk_start)

    if full_prices.empty:
        msg = "Błąd pobierania danych z Yahoo (rate limit / brak połączenia?) — spróbuj ponownie za chwilę."
        return [], html.Div(msg, style={"color": THEME["orange"]}), empty_fig, empty_fig, empty_fig, meta_info, []

    # RAW (bez dropna!) -- ragged multi-exchange NaN-y są potrzebne compute_crash_overlap_matrix
    # (Option A, per-para przecięcie dat); compute_sandbox_allocation robi lokalny .dropna()
    # tylko tam, gdzie faktycznie potrzebny jest wspólny indeks (Sigma_eps).
    risk_prices_full = full_prices[[t for t in tickers if t in full_prices.columns]].tail(1260)
    prices = full_prices[full_prices.index >= created_at]

    if prices.empty or len(prices) < 2:
        msg = "Za mało sesji giełdowych od dnia zapisu, żeby narysować krzywą equity (za świeży snapshot — wróć za dzień/dwa)."
        return [], html.Div(msg, style={"color": THEME["orange"]}), empty_fig, empty_fig, empty_fig, meta_info, []

    missing_tickers = [t for t in tickers if t not in prices.columns]
    stock_prices = prices[[t for t in tickers if t in prices.columns]].dropna(how="any")
    if stock_prices.shape[1] == 0 or len(stock_prices) < 2:
        return [], html.Div("Brak wspólnych danych cenowych dla żadnej spółki z portfela.", style={"color": THEME["orange"]}), empty_fig, empty_fig, empty_fig, meta_info, []

    valid_tickers = list(stock_prices.columns)
    stock_returns = stock_prices.pct_change().fillna(0.0)

    manual_weights, sandbox_note, sandbox_extra = compute_sandbox_allocation(
        record, valid_tickers, risk_prices_full, sb_lambda, sb_gamma, sb_kappa, sb_wmax, sb_rf, sb_nref,
        sb_alpha if isinstance(sb_alpha, (int, float)) else 0.5
    )

    # --- EQUITY CURVES (proste zwroty -- jedyny poprawny sposób na kompoundowanie do V_t) ---
    w_orig_vec = orig_weights.reindex(valid_tickers).fillna(0.0).values
    if w_orig_vec.sum() > 1e-9:
        w_orig_vec = w_orig_vec / w_orig_vec.sum()
    orig_daily_ret = pd.Series(stock_returns.values @ w_orig_vec, index=stock_returns.index)
    orig_eq = 100.0 * (1.0 + orig_daily_ret).cumprod()

    w_sb_vec = manual_weights.reindex(valid_tickers).fillna(0.0).values
    sb_daily_ret = pd.Series(stock_returns.values @ w_sb_vec, index=stock_returns.index)
    sb_eq = 100.0 * (1.0 + sb_daily_ret).cumprod()

    w_eq_vec = np.full(len(valid_tickers), 1.0 / len(valid_tickers))
    eq_1n_eq = 100.0 * (1.0 + pd.Series(stock_returns.values @ w_eq_vec, index=stock_returns.index)).cumprod()

    vols = stock_returns.std().replace(0, np.nan)
    inv_vols = 1.0 / vols
    w_invvol_vec = (inv_vols / inv_vols.sum()).fillna(1.0 / len(valid_tickers)).values
    invvol_eq = 100.0 * (1.0 + pd.Series(stock_returns.values @ w_invvol_vec, index=stock_returns.index)).cumprod()

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=orig_eq.index, y=orig_eq.values, mode='lines', name="Original Portfolio (zapisany)", line=dict(color=THEME["accent"], width=3)))
    fig_eq.add_trace(go.Scatter(x=sb_eq.index, y=sb_eq.values, mode='lines', name="Manual Sandbox (suwaki)", line=dict(color=THEME["orange"], width=2.5, dash='dash')))
    if "1N" in (benchmarks or []):
        fig_eq.add_trace(go.Scatter(x=eq_1n_eq.index, y=eq_1n_eq.values, mode='lines', name="Equal Weight (1/N)", line=dict(color="#4A90A4", width=2)))
    if "INV_VOL" in (benchmarks or []):
        fig_eq.add_trace(go.Scatter(x=invvol_eq.index, y=invvol_eq.values, mode='lines', name="Equal Risk (Inv-Vol)", line=dict(color=THEME["pos"], width=2)))
    if "SPY" in (benchmarks or []) and "SPY" in prices.columns:
        spy_s = prices["SPY"].dropna()
        spy_eq = 100.0 * (1.0 + spy_s.pct_change().fillna(0.0)).cumprod()
        fig_eq.add_trace(go.Scatter(x=spy_eq.index, y=spy_eq.values, mode='lines', name="S&P 500 (SPY)", line=dict(color="#888888", width=1.5)))
    if "QQQ" in (benchmarks or []) and "QQQ" in prices.columns:
        qqq_s = prices["QQQ"].dropna()
        qqq_eq = 100.0 * (1.0 + qqq_s.pct_change().fillna(0.0)).cumprod()
        fig_eq.add_trace(go.Scatter(x=qqq_eq.index, y=qqq_eq.values, mode='lines', name="Nasdaq 100 (QQQ)", line=dict(color="#AA66FF", width=1.5)))
    fig_eq.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"],
        margin=dict(l=40, r=20, t=20, b=30), font_family=THEME["font"],
        yaxis=dict(title="Portfolio Value (Base 100)", gridcolor="#1E1E28", side="right"),
        xaxis=dict(gridcolor="#1E1E28"), showlegend=True, legend=dict(orientation="h", y=1.18, font=dict(color=THEME["text_white"], size=10))
    )

    asset_perf = (stock_prices.iloc[-1] / stock_prices.iloc[0] - 1.0) * 100.0
    asset_perf = asset_perf.sort_values(ascending=False)
    colors_bar = [THEME["pos"] if v >= 0 else THEME["neg"] for v in asset_perf.values]
    fig_bar = go.Figure(go.Bar(x=asset_perf.index, y=asset_perf.values, marker_color=colors_bar))
    fig_bar.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"],
        margin=dict(l=30, r=10, t=10, b=30), font_family=THEME["font"],
        yaxis=dict(ticksuffix="%", gridcolor="#1E1E28"), xaxis=dict(showgrid=False, tickfont=dict(size=9))
    )

    orig_dd = (orig_eq - orig_eq.cummax()) / orig_eq.cummax() * 100.0
    sb_dd = (sb_eq - sb_eq.cummax()) / sb_eq.cummax() * 100.0
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=orig_dd.index, y=orig_dd.values, mode='lines', name="Original DD", line=dict(color=THEME["accent"], width=1.5)))
    fig_dd.add_trace(go.Scatter(x=sb_dd.index, y=sb_dd.values, mode='lines', name="Sandbox DD", line=dict(color=THEME["orange"], width=1.5)))
    fig_dd.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"],
        margin=dict(l=30, r=10, t=10, b=30), font_family=THEME["font"],
        yaxis=dict(ticksuffix="%", gridcolor="#1E1E28"), xaxis=dict(gridcolor="#1E1E28"), showlegend=False
    )

    promoted_set = set(sandbox_extra.get("promoted_tickers", []))
    cluster_of_sb = sandbox_extra.get("cluster_of", {})
    mu_vec_sb = sandbox_extra.get("mu_vec", {})

    tbl_rows = []
    for t in valid_tickers:
        orig_w = orig_weights.get(t, 0.0)
        sb_w = manual_weights.get(t, 0.0)
        ret = asset_perf.get(t, 0.0)
        cluster_label = cluster_of_sb.get(t, "—")
        cluster_display = f"★ {cluster_label}" if t in promoted_set else cluster_label
        tbl_rows.append({
            "Ticker": t,
            "Cluster": cluster_display,
            "Original": orig_w * 100.0,
            "Sandbox": sb_w * 100.0,
            "Delta": (sb_w - orig_w) * 100.0,
            "MuI": mu_vec_sb.get(t, 0.0) * 100.0,
            "Return": ret,
            "Contribution": sb_w * ret,
        })
    tbl_rows.sort(key=lambda r: r["Sandbox"], reverse=True)

    weights_table = dash_table.DataTable(
        columns=[
            {"name": "Ticker", "id": "Ticker"},
            {"name": "Cluster (Sandbox)", "id": "Cluster"},
            {"name": "Original Weight", "id": "Original", "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": "Sandbox Weight", "id": "Sandbox", "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": "Δ (pp)", "id": "Delta", "type": "numeric", "format": {"specifier": "+.1f"}},
            {"name": "μ_i (sandbox)", "id": "MuI", "type": "numeric", "format": {"specifier": "+.1f"}},
            {"name": "Return Since Entry", "id": "Return", "type": "numeric", "format": {"specifier": "+.1f"}},
            {"name": "Contribution to Return", "id": "Contribution", "type": "numeric", "format": {"specifier": "+.2f"}},
        ],
        data=tbl_rows, page_size=15, sort_action='native',
        style_header=datatable_style_header(),
        style_data=datatable_style_data(),
        style_cell=datatable_style_cell(),
        style_cell_conditional=[{'if': {'column_id': 'Ticker'}, 'fontWeight': 'bold', 'textAlign': 'left', 'color': THEME['accent']}],
        style_data_conditional=[
            datatable_row_alt_rule(),
            {'if': {'filter_query': '{Cluster} contains "★"', 'column_id': 'Cluster'}, 'color': THEME['warn'], 'fontWeight': 'bold'},
            {'if': {'filter_query': '{Delta} > 0', 'column_id': 'Delta'}, 'color': THEME['pos']},
            {'if': {'filter_query': '{Delta} < 0', 'column_id': 'Delta'}, 'color': THEME['neg']},
            {'if': {'filter_query': '{Return} > 0', 'column_id': 'Return'}, 'color': THEME['pos']},
            {'if': {'filter_query': '{Return} < 0', 'column_id': 'Return'}, 'color': THEME['neg']},
            {'if': {'filter_query': '{Contribution} > 0', 'column_id': 'Contribution'}, 'color': THEME['pos']},
            {'if': {'filter_query': '{Contribution} < 0', 'column_id': 'Contribution'}, 'color': THEME['neg']},
            {'if': {'filter_query': '{Sandbox} = 0', 'column_id': 'Sandbox'}, 'color': THEME['text_dim']},
        ]
    )
    weights_children = [weights_table]
    if sandbox_note:
        weights_children.append(html.Div(f"ℹ {sandbox_note}", style={"color": THEME["text_dim"], "fontSize": "10px", "marginTop": "10px"}))
    if missing_tickers:
        weights_children.append(html.Div(f"Brak danych live dla: {', '.join(missing_tickers)}", style={"color": THEME["orange"], "fontSize": "10px", "marginTop": "6px"}))

    tot_orig_ret = (orig_eq.iloc[-1] / 100.0 - 1.0) * 100.0
    tot_eq_ret = (eq_1n_eq.iloc[-1] / 100.0 - 1.0) * 100.0
    alpha_vs_1n = tot_orig_ret - tot_eq_ret
    orig_vol_ann = float(orig_daily_ret.std() * np.sqrt(252) * 100.0)

    alpha_vs_spy_str = "N/A"
    if "SPY" in prices.columns:
        spy_s = prices["SPY"].dropna()
        if len(spy_s) >= 2:
            spy_tot_ret = (spy_s.iloc[-1] / spy_s.iloc[0] - 1.0) * 100.0
            alpha_vs_spy_str = f"{(tot_orig_ret - spy_tot_ret):+.2f}%"

    kpi_cards = build_kpi_strip([
        {"label": "PORTFOLIO RETURN", "value": f"{tot_orig_ret:+.2f}%", "sub": f"Since {created_at}", "color": THEME["pos"] if tot_orig_ret >= 0 else THEME["neg"]},
        {"label": "ANNUALIZED VOLATILITY", "value": f"{orig_vol_ann:.2f}%", "sub": "Original portfolio"},
        {"label": "MAX DRAWDOWN", "value": f"{orig_dd.min():.2f}%", "sub": "Peak-to-trough", "color": THEME["neg"]},
        {"label": "ALPHA vs 1/N", "value": f"{alpha_vs_1n:+.2f}%", "sub": "Pure weighting edge", "color": THEME["accent"] if alpha_vs_1n >= 0 else THEME["neg"]},
        {"label": "ALPHA vs SPY", "value": alpha_vs_spy_str, "sub": "Same holding period"},
    ])

    # Mini KPI sandboxa -- metryki portfela POD BIEŻĄCYMI suwakami (parytet z KPI Tab 4),
    # osobne od kpi_cards powyżej (te opisują zapisany, oryginalny portfel od dnia zapisu).
    def _fmt_pct(v):
        return f"{v*100:+.1f}%" if isinstance(v, (int, float)) else "N/A"
    def _fmt_num(v):
        return f"{v:.2f}" if isinstance(v, (int, float)) else "N/A"

    mini_kpi = build_kpi_strip([
        {"label": "μ_P (sandbox)", "value": _fmt_pct(sandbox_extra.get("mu_p")), "sub": "Expected return"},
        {"label": "δ_P (sandbox)", "value": _fmt_pct(sandbox_extra.get("delta_p")), "sub": "Downside risk"},
        {"label": "TPS_P (sandbox)", "value": _fmt_num(sandbox_extra.get("tps_p")), "sub": "Tail-Penalized Sortino", "color": THEME["accent"]},
    ])

    return kpi_cards, html.Div(weights_children), fig_eq, fig_bar, fig_dd, meta_info, mini_kpi