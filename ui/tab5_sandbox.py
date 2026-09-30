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
from scipy.stats import gaussian_kde
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State

from ui.app_instance import app, cache
from ui.theme import THEME, CHART_COLORS
from ui.components import build_kpi_card, build_kpi_strip, datatable_style_header, datatable_style_cell, datatable_style_data, datatable_row_alt_rule, build_pair_overlay_output, build_tps_formula_breakdown
from engine.risk import compute_estrada_matrix, compute_crash_overlap_matrix
from engine.optimizer import run_optimization_with_singleton_split
from engine.returns import compute_composite_upside_row
from engine.pairs import find_matched_pairs_for_overlay, apply_tilts_to_matched_pairs
from engine.sobol_analysis import (
    run_morris_screen, run_sobol_batch, estimate_run_cost, PARAM_ORDER, DEFAULT_PARAM_RANGES, HORIZON_DAYS,
)
from data import snapshot_store as snap
from data.market_data import fetch_universe_prices, fetch_treasury_yield_on_date

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
    Output("dropdown-sobol-snapshot", "options"), Output("dropdown-sobol-snapshot", "value"),
    Input("store-snapshots-refresh", "data"),
)
def load_snapshot_list(_):
    snapshots = snap.list_snapshots()
    options = [{"label": f"{s['snapshot_name']}  ({s['snapshot_id']})", "value": s["snapshot_id"]} for s in snapshots]
    value = options[0]["value"] if options else None
    return options, value, options, value


@app.callback(
    Output("slider-sb-rf", "value"), Output("sb-rf-autofetch-note", "children"),
    Input("dropdown-snapshot-select", "value"),
    prevent_initial_call=True
)
def autofetch_sandbox_rf(snapshot_id):
    """
    Confirmed 2026-09-28 (Etap 8h): auto-wypełnia R_f/hurdle rate w Sandboxie
    rzeczywistą rentownością 10Y (^TNX) NA DZIEŃ UTWORZENIA analizowanego
    zapisu -- zamiast statycznego 0.045 -- bo analizując stary snapshot
    właściciel projektu nie pamięta/nie zna z pamięci, jaka była wtedy
    stopa. Na głównej zakładce Rebalance (input-rf) NIC się nie zmienia --
    tam portfel powstaje "dziś", więc ręczne wejście dzisiejszej wartości
    zostaje.

    Suwak zostaje suwakiem -- to tylko wypełnia jego wartość PO wyborze
    zapisu, właściciel projektu może ją dalej ręcznie nadpisać w dowolnym
    momencie (zgodnie z ustaloną zasadą "dane manualne" -- to wygoda z
    sensownym domyślnym punktem startowym, nie coś co cicho nadpisuje jego
    decyzję).

    Celowo NIE dotyka zakresu przeszukiwania `rf` w gridzie zakładki ANALIZA
    SOBOLA (sobol-range-rf-min/max, engine.sobol_analysis.DEFAULT_PARAM_RANGES)
    -- to osobna, świadomie szeroka, płaska metodyka sweepu wrażliwości
    (patrz Etap 8g), a nie coś co powinno się cicho przesuwać per-snapshot.

    Nie dotyka W OGÓLE definicji Sortino w engine/sobol_analysis.py -- ta
    zostaje na sztywno MAR=0.0, celowo niepowiązana z R_f (Etap 8b).
    """
    if not snapshot_id:
        return dash.no_update, ""

    record = snap.get_snapshot(snapshot_id)
    if not record:
        return dash.no_update, ""

    created_at_str = (record.get("created_at") or "")[:10]
    if not created_at_str:
        return dash.no_update, ""

    yield_frac = fetch_treasury_yield_on_date(created_at_str)
    if yield_frac is None:
        return dash.no_update, f"⚠ Nie udało się pobrać ^TNX na dzień {created_at_str} -- zostaje poprzednia/domyślna wartość, ustaw ręcznie jeśli chcesz."
    return round(yield_frac, 4), f"✓ Auto: rentowność 10Y (^TNX) na dzień utworzenia zapisu ({created_at_str}): {yield_frac*100:.2f}% -- można nadpisać ręcznie."


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


def compute_sandbox_allocation(record, tickers, risk_prices, sb_lambda, sb_gamma, sb_kappa, sb_wmax, sb_rf, sb_nref, sb_alpha, sb_nu=0.0):
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
                    mu_vec.reindex(usable_tickers), sigma_sb, crash_sb["K"], clusters_dict, sb_lambda, w_max=sb_wmax, Rf=sb_rf, nu=sb_nu
                )
                w_raw = split_result["final"]["weights"].reindex(tickers).fillna(0.0)
                extra["cluster_of"] = {t: ck for ck, members in split_result["final_clusters_dict"].items() for t in members}
                extra["promoted_tickers"] = split_result["promoted_tickers"]
                extra["mu_p"] = split_result["final"]["mu_p"]
                extra["delta_p"] = split_result["final"]["delta_p"]
                extra["k_penalty_p"] = split_result["final"]["k_penalty_p"]
                extra["tps_p"] = split_result["final"]["tps_p"]
                extra["lam_used"] = sb_lambda
                extra["nu_used"] = sb_nu
                extra["rf_used"] = sb_rf
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


# Confirmed 2026-09-28: pairs each Sandbox slider with the small "własna wartość"
# dcc.Input added beside it in ui/layout.py (_slider_custom_input). A dcc.Slider
# genuinely cannot be dragged past its own min/max -- unlike Rebalance's
# STAGE4A_PARAMS_CONFIG cards, which are plain dcc.Input number fields with no
# HTML min/max and no server-side clamping, so a value outside the "Recommended"
# badge already just works there today, no change needed.
#
# Deliberately widens the SLIDER's own min/max (never shrinks them back) and
# sets its value, rather than threading a separate "custom override" value
# through update_forward_tracker/run_sobol_analysis/etc. -- every existing
# callback in this file already reads these sliders via Input/State on their
# "value" prop, so the slider stays the one source of truth and nothing
# downstream needed to change.
_SANDBOX_SLIDER_IDS = ["slider-sb-alpha", "slider-sb-lambda", "slider-sb-nu", "slider-sb-gamma",
                        "slider-sb-kappa", "slider-sb-wmax", "slider-sb-rf", "slider-sb-nref"]


@app.callback(
    # allow_duplicate=True on every Output here: slider-sb-rf.value is ALSO
    # an Output of autofetch_sandbox_rf (Etap 8h, fires on snapshot change) --
    # Dash refuses two callbacks writing the same Output otherwise ("Output 6
    # (slider-sb-rf.value) is already in use", hit 2026-09-28). Set on all 24
    # outputs uniformly rather than just the one that currently collides, so
    # a future slider-writing callback doesn't hit this again unnoticed.
    [Output(sid, "value", allow_duplicate=True) for sid in _SANDBOX_SLIDER_IDS] +
    [Output(sid, "min", allow_duplicate=True) for sid in _SANDBOX_SLIDER_IDS] +
    [Output(sid, "max", allow_duplicate=True) for sid in _SANDBOX_SLIDER_IDS],
    [Input(f"custom-{sid}", "value") for sid in _SANDBOX_SLIDER_IDS],
    [State(sid, "min") for sid in _SANDBOX_SLIDER_IDS] + [State(sid, "max") for sid in _SANDBOX_SLIDER_IDS],
    prevent_initial_call=True,
)
def apply_custom_sandbox_slider_value(*args):
    n = len(_SANDBOX_SLIDER_IDS)
    custom_values, mins, maxs = args[:n], args[n:2 * n], args[2 * n:3 * n]

    trigger_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0] if dash.callback_context.triggered else None
    values_out, mins_out, maxs_out = [dash.no_update] * n, [dash.no_update] * n, [dash.no_update] * n

    if trigger_id and trigger_id.startswith("custom-"):
        base_id = trigger_id[len("custom-"):]
        if base_id in _SANDBOX_SLIDER_IDS:
            idx = _SANDBOX_SLIDER_IDS.index(base_id)
            val = custom_values[idx]
            if isinstance(val, (int, float)) and np.isfinite(val):
                values_out[idx] = val
                mins_out[idx] = min(mins[idx], val) if isinstance(mins[idx], (int, float)) else val
                maxs_out[idx] = max(maxs[idx], val) if isinstance(maxs[idx], (int, float)) else val

    return values_out + mins_out + maxs_out


@app.callback(
    Output("kpi-summary-row", "children"), Output("sandbox-weights-table-container", "children"),
    Output("graph-forward-equity-curves", "figure"), Output("graph-asset-returns-bar", "figure"),
    Output("graph-forward-drawdowns", "figure"), Output("snapshot-meta-info", "children"),
    Output("sandbox-mini-kpi-row", "children"), Output("store-sandbox-manual-weights", "data"),
    Output("sandbox-formula-breakdown", "children"),
    Input("dropdown-snapshot-select", "value"), Input("slider-sb-alpha", "value"), Input("slider-sb-lambda", "value"), Input("slider-sb-nu", "value"), Input("slider-sb-gamma", "value"),
    Input("slider-sb-kappa", "value"), Input("slider-sb-rf", "value"), Input("slider-sb-nref", "value"),
    Input("slider-sb-wmax", "value"), Input("checklist-benchmarks", "value"), Input("btn-refresh-live", "n_clicks"),
    Input("toggle-sandbox-pair-overlay", "value"), Input("input-sandbox-overlay-theta", "value"), Input("store-sandbox-pair-overlay-match-cache", "data"),
    prevent_initial_call=True
)
def update_forward_tracker(snapshot_id, sb_alpha, sb_lambda, sb_nu, sb_gamma, sb_kappa, sb_rf, sb_nref, sb_wmax, benchmarks, _,
                            overlay_toggle_value, overlay_theta, pair_overlay_cache):
    overlay_enabled = "ON" in (overlay_toggle_value or [])
    empty_fig = go.Figure()
    empty_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"])
    empty_msg_style = {"color": THEME["text_dim"], "fontSize": "12px"}

    if not snapshot_id:
        return [], html.Div("Wybierz zapisany portfel z listy powyżej.", style=empty_msg_style), empty_fig, empty_fig, empty_fig, "", [], {}, html.Div()

    record = snap.get_snapshot(snapshot_id)
    if not record:
        return [], html.Div("Nie znaleziono snapshotu (mógł zostać usunięty).", style={"color": THEME["orange"]}), empty_fig, empty_fig, empty_fig, "", [], {}, html.Div()

    created_at_full = record.get("created_at", "")
    created_at = created_at_full[:10] if created_at_full else "?"
    holding_days = snap.holding_days_since(record)
    meta_info_text = f"Utworzono: {created_at}  •  {holding_days} dni w portfolio"

    # Confirmed 2026-09-26: odznaka statusu nakładki RV zapisanej PRZY ZAPISIE tego
    # snapshotu (data/snapshot_store.py "rv_overlay") -- ma pokazać, czy to co jest
    # w pliku pochodzi z ręcznie zweryfikowanej theta, czy z niezwalidowanej wartości
    # domyślnej użytej automatycznie, bo użytkownik nie klikał "ZNAJDŹ PARY" przy
    # zapisie. Snapshoty sprzed tej funkcji (klucz nieobecny) nie pokazują nic --
    # zero zmiany zachowania dla starych plików. To jest CZYSTO INFORMACYJNE: nie
    # steruje wcale niezależnym, żywym przełącznikiem "toggle-sandbox-pair-overlay"
    # poniżej, który dalej liczy swoją własną nakładkę na bieżąco.
    rv_overlay = record.get("rv_overlay")
    overlay_badge = None
    if rv_overlay and rv_overlay.get("computed"):
        n_pairs = len(rv_overlay.get("matched_pairs", []))
        if rv_overlay.get("theta_source") == "user_reviewed":
            overlay_badge = html.Span(f"  •  ✓ Nakładka RV przy zapisie: {n_pairs} par, θ={rv_overlay.get('theta')} (zweryfikowana)", style={"color": THEME["accent"]})
        else:
            overlay_badge = html.Span(f"  •  ⚠ Nakładka RV przy zapisie: {n_pairs} par, θ={rv_overlay.get('theta')} (domyślna, tymczasowa)", style={"color": THEME["orange"]})
    elif rv_overlay and not rv_overlay.get("computed"):
        overlay_badge = html.Span(f"  •  ⚠ Nakładka RV nie policzona przy zapisie ({rv_overlay.get('reason', '?')})", style={"color": THEME["text_dim"]})

    meta_info = [meta_info_text, overlay_badge] if overlay_badge else meta_info_text

    orig_weights = pd.Series(record.get("final_weights", {}))
    tickers = list(orig_weights.index)
    if not tickers:
        return [], html.Div("Snapshot nie zawiera żadnych aktywów.", style={"color": THEME["orange"]}), empty_fig, empty_fig, empty_fig, meta_info, [], {}, html.Div()

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
        return [], html.Div(msg, style={"color": THEME["orange"]}), empty_fig, empty_fig, empty_fig, meta_info, [], {}, html.Div()

    # RAW (bez dropna!) -- ragged multi-exchange NaN-y są potrzebne compute_crash_overlap_matrix
    # (Option A, per-para przecięcie dat); compute_sandbox_allocation robi lokalny .dropna()
    # tylko tam, gdzie faktycznie potrzebny jest wspólny indeks (Sigma_eps).
    risk_prices_full = full_prices[[t for t in tickers if t in full_prices.columns]].tail(1260)
    prices = full_prices[full_prices.index >= created_at]

    if prices.empty or len(prices) < 2:
        msg = "Za mało sesji giełdowych od dnia zapisu, żeby narysować krzywą equity (za świeży snapshot — wróć za dzień/dwa)."
        return [], html.Div(msg, style={"color": THEME["orange"]}), empty_fig, empty_fig, empty_fig, meta_info, [], {}, html.Div()

    missing_tickers = [t for t in tickers if t not in prices.columns]
    stock_prices = prices[[t for t in tickers if t in prices.columns]].dropna(how="any")
    if stock_prices.shape[1] == 0 or len(stock_prices) < 2:
        return [], html.Div("Brak wspólnych danych cenowych dla żadnej spółki z portfela.", style={"color": THEME["orange"]}), empty_fig, empty_fig, empty_fig, meta_info, [], {}, html.Div()

    valid_tickers = list(stock_prices.columns)
    stock_returns = stock_prices.pct_change().fillna(0.0)

    manual_weights, sandbox_note, sandbox_extra = compute_sandbox_allocation(
        record, valid_tickers, risk_prices_full, sb_lambda, sb_gamma, sb_kappa, sb_wmax, sb_rf, sb_nref,
        sb_alpha if isinstance(sb_alpha, (int, float)) else 0.5, sb_nu=(sb_nu if isinstance(sb_nu, (int, float)) else 0.0)
    )
    manual_weights_raw = manual_weights.to_dict()

    # Confirmed 2026-09-19 (poprawka na prosbe wlasciciela projektu): NIE
    # zamieniamy juz krzywej "Manual Sandbox" w miejscu -- to ukrywalo baze
    # do porownania. Zamiast tego, gdy przelacznik WLACZONY i dobor par juz
    # zrobiony, liczymy OSOBNY wektor wag (manual_weights_overlay) i OSOBNA
    # krzywa equity, ktora pokazuje sie OBOK oryginalnej "Manual Sandbox"
    # (nie zamiast niej) -- uzytkownik widzi obie na raz, bezposrednio
    # porownywalne. Baza (manual_weights) zostaje NIETKNIETA w kazdym
    # przypadku.
    manual_weights_overlay = None
    if overlay_enabled and pair_overlay_cache and pair_overlay_cache.get("matched_pairs_info"):
        overlay_theta = overlay_theta if isinstance(overlay_theta, (int, float)) else 0.15
        tilt_result = apply_tilts_to_matched_pairs(manual_weights_raw, pair_overlay_cache["matched_pairs_info"], theta=overlay_theta, wmax=sb_wmax)
        manual_weights_overlay = pd.Series(tilt_result["weights"])

    # --- EQUITY CURVES (proste zwroty -- jedyny poprawny sposób na kompoundowanie do V_t) ---
    w_orig_vec = orig_weights.reindex(valid_tickers).fillna(0.0).values
    if w_orig_vec.sum() > 1e-9:
        w_orig_vec = w_orig_vec / w_orig_vec.sum()
    orig_daily_ret = pd.Series(stock_returns.values @ w_orig_vec, index=stock_returns.index)
    orig_eq = 100.0 * (1.0 + orig_daily_ret).cumprod()

    w_sb_vec = manual_weights.reindex(valid_tickers).fillna(0.0).values
    sb_daily_ret = pd.Series(stock_returns.values @ w_sb_vec, index=stock_returns.index)
    sb_eq = 100.0 * (1.0 + sb_daily_ret).cumprod()

    sb_overlay_eq = None
    if manual_weights_overlay is not None:
        w_sb_overlay_vec = manual_weights_overlay.reindex(valid_tickers).fillna(0.0).values
        sb_overlay_daily_ret = pd.Series(stock_returns.values @ w_sb_overlay_vec, index=stock_returns.index)
        sb_overlay_eq = 100.0 * (1.0 + sb_overlay_daily_ret).cumprod()

    w_eq_vec = np.full(len(valid_tickers), 1.0 / len(valid_tickers))
    eq_1n_eq = 100.0 * (1.0 + pd.Series(stock_returns.values @ w_eq_vec, index=stock_returns.index)).cumprod()

    vols = stock_returns.std().replace(0, np.nan)
    inv_vols = 1.0 / vols
    w_invvol_vec = (inv_vols / inv_vols.sum()).fillna(1.0 / len(valid_tickers)).values
    invvol_eq = 100.0 * (1.0 + pd.Series(stock_returns.values @ w_invvol_vec, index=stock_returns.index)).cumprod()

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=orig_eq.index, y=orig_eq.values, mode='lines', name="Original Portfolio (zapisany)", line=dict(color=THEME["accent"], width=3)))
    fig_eq.add_trace(go.Scatter(x=sb_eq.index, y=sb_eq.values, mode='lines', name="Manual Sandbox (suwaki)", line=dict(color=THEME["orange"], width=2.5, dash='dash')))
    if sb_overlay_eq is not None:
        fig_eq.add_trace(go.Scatter(x=sb_overlay_eq.index, y=sb_overlay_eq.values, mode='lines', name="Manual Sandbox + RV overlay", line=dict(color="#00C853", width=2.5, dash='dot')))
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

    formula_breakdown = build_tps_formula_breakdown(
        mu_p=sandbox_extra.get("mu_p"), rf=sandbox_extra.get("rf_used", sb_rf if isinstance(sb_rf, (int, float)) else 0.045),
        sigma_p=sandbox_extra.get("delta_p"), k_p=sandbox_extra.get("k_penalty_p"),
        lam=sandbox_extra.get("lam_used", 0.0), nu=sandbox_extra.get("nu_used", 0.0),
        tps_p=sandbox_extra.get("tps_p"),
    )

    return kpi_cards, html.Div(weights_children), fig_eq, fig_bar, fig_dd, meta_info, mini_kpi, manual_weights_raw, formula_breakdown

# ---------------------------------------------------------------------------
# Relative Value -- nakladka po optymalizacji w Sandboxie (2026-09-18),
# ta sama dwuetapowa architektura co Tab 4 (patrz komentarz modulu w
# ui/tab4_rebalance.py dla pelnego uzasadnienia projektowego). Roznica:
# tutaj przelacznik `toggle-sandbox-pair-overlay` faktycznie WPLYWA na
# sledzona krzywa equity (patrz update_forward_tracker powyzej), nie tylko
# na osobna tabele diagnostyczna -- ma to sens w Sandboxie, ktorego calym
# celem jest porownywanie wplywu roznych wyborow na sledzony wynik.
# ---------------------------------------------------------------------------

@app.callback(
    Output("store-sandbox-pair-overlay-match-cache", "data"), Output("sandbox-pair-overlay-status", "children", allow_duplicate=True),
    Input("btn-sandbox-run-pair-overlay", "n_clicks"),
    State("input-sandbox-overlay-theta", "value"), State("store-sandbox-manual-weights", "data"),
    State("dropdown-snapshot-select", "value"),
    background=True, progress=[Output("sandbox-pair-overlay-status", "children", allow_duplicate=True)],
    prevent_initial_call=True,
)
def find_sandbox_pair_overlay_matches(set_progress, _n_clicks, theta, manual_weights_raw, snapshot_id):
    """
    Etap A (drogi) dla Sandboxa -- patrz komentarz modulu powyzej.

    KRYTYCZNA POPRAWKA (2026-09-19, confirmed przez wlasciciela projektu):
    poprzednia wersja pobierala ceny "period=10y konczace sie DZISIAJ",
    niezaleznie od tego, kiedy snapshot zostal faktycznie utworzony -- to
    byl prawdziwy look-ahead bias, bezposrednio lamiacy wlasna zasade
    projektu z Part I Sekcja 6.2 ("walk-forward protocol... entirely free
    of look-ahead and survivorship biases"). Jesli portfel zostal zapisany
    1 wrzesnia, analiza doboru par i Z-score NIE MOZE widziec zadnych cen
    z 2, 3, ..., 17 wrzesnia -- decyzja "ktore pary i jak przechylic" musi
    byc mozliwa do podjecia WYLACZNIE na podstawie danych dostepnych W DNIU
    utworzenia snapshotu, dokladnie tak samo jak oryginalne wagi solvera
    (final_weights) same w sobie sa zamrozonym zapisem z tamtego dnia.

    Naprawa: pobieramy jak dotychczas (do dzisiaj, bo tyle danych jest
    fizycznie dostepnych), ale NATYCHMIAST obcinamy do
    `prices_df.index <= created_at` PRZED przekazaniem czegokolwiek do
    find_matched_pairs_for_overlay -- test trwalosci, dobor par, i Z-score
    "aktualny" (ktory w tym kontekscie oznacza "aktualny NA DZIEN
    utworzenia snapshotu", nie "aktualny dzisiaj") licza sie odtad
    wylacznie na historii sprzed/do dnia zapisu. Wynikowe przechylenie jest
    wiec decyzja mozliwa do podjecia W TAMTYM MOMENCIE -- a to, jak dobrze
    ta decyzja wypada, sledzimy juz normalnie (Manual Sandbox + RV overlay)
    naprzod, na realnych, poznniejszych, nie uzytych do analizy cenach.
    """
    theta = theta or 0.15

    if not manual_weights_raw:
        return dash.no_update, html.Div("Brak wag Sandboxa -- wybierz snapshot i poczekaj, aż suwaki się przeliczą.", style={"color": THEME["orange"]})

    selected_tickers = [t for t, w in manual_weights_raw.items() if w and w > 1e-9]
    if len(selected_tickers) < 2:
        return dash.no_update, html.Div("Za mało spółek z dodatnią wagą do sprawdzenia par.", style={"color": THEME["orange"]})

    record = snap.get_snapshot(snapshot_id) if snapshot_id else None
    if not record:
        return dash.no_update, html.Div("Brak wybranego snapshotu -- wybierz portfel z listy powyżej.", style={"color": THEME["orange"]})
    created_at_str = (record.get("created_at") or "")[:10]
    try:
        created_at_ts = pd.Timestamp(created_at_str)
    except (ValueError, TypeError):
        return dash.no_update, html.Div("Snapshot nie ma poprawnej daty utworzenia -- nie mogę bezpiecznie odciąć danych.", style={"color": THEME["orange"]})

    set_progress([f"Pobieram niezależnie 10 lat historii cen dla {len(selected_tickers)} wybranych spółek..."])
    prices_df_full, valid_tickers = fetch_universe_prices(selected_tickers, period="10y")
    if prices_df_full.empty or len(valid_tickers) < 2:
        return dash.no_update, html.Div("Nie udało się pobrać wystarczająco długiej (10-letniej) historii cenowej.", style={"color": THEME["orange"]})

    # KLUCZOWE OBCIECIE -- bez tego caly dobor par widzialby przyszlosc wzgledem dnia zapisu snapshotu
    prices_df = prices_df_full[prices_df_full.index <= created_at_ts]
    if prices_df.empty:
        return dash.no_update, html.Div(f"Brak jakichkolwiek danych cenowych sprzed daty utworzenia snapshotu ({created_at_str}).", style={"color": THEME["orange"]})

    def _report_progress(label, i, total):
        if label == "Etap 1: tanie sito":
            set_progress([f"Tanie sito theta (kanonizacja): kombinacja {i}/{total}..."])
        else:
            set_progress([f"Pełny test trwałości ({label}): para {i}/{total}..."])

    set_progress([f"Sprawdzam pary wśród {len(valid_tickers)} spółek z dodatnią wagą, dane wyłącznie do {created_at_str}..."])
    match_data = find_matched_pairs_for_overlay(manual_weights_raw, prices_df, theta=theta, on_progress=_report_progress)
    match_data["as_of_date"] = created_at_str
    status = html.Div(
        f"Dobór par zakończony (dane wyłącznie do {created_at_str}, dnia utworzenia snapshotu -- bez zaglądania w przyszłość) -- "
        f"{match_data['n_selected']} spółek, {match_data['n_qualifying']} par kwalifikujących się, "
        f"{match_data['n_eligible']} dopuszczalnych, {len(match_data['matched_pairs_info'])} dopasowanych. "
        f"Zaznacz przełącznik powyżej, żeby zastosować na śledzonej krzywej.",
        style={"color": THEME["accent"]}
    )
    return match_data, status


@app.callback(
    Output("sandbox-pair-overlay-results", "children"), Output("sandbox-pair-overlay-status", "children", allow_duplicate=True),
    Input("store-sandbox-pair-overlay-match-cache", "data"), Input("input-sandbox-overlay-theta", "value"),
    State("store-sandbox-manual-weights", "data"), State("slider-sb-wmax", "value"),
    prevent_initial_call=True,
)
def render_sandbox_pair_overlay_tilts(match_cache, theta, manual_weights_raw, sb_wmax):
    """Etap B (tani) dla Sandboxa -- reaguje na kazda zmiane thety LUB na
    swiezy wynik Etapu A, zero pobierania danych."""
    theta = theta or 0.15
    sb_wmax = sb_wmax if isinstance(sb_wmax, (int, float)) and sb_wmax > 0 else 0.30
    if not manual_weights_raw:
        return dash.no_update, dash.no_update
    return build_pair_overlay_output(manual_weights_raw, match_cache, theta, sb_wmax, apply_tilts_to_matched_pairs, today_label="dzień zapisu")


# ---------------------------------------------------------------------------
# Zakładka SOBOL -- globalna analiza wrażliwości (2026-09-19). Działa na
# JEDNYM, zamrożonym zapisie i JEDNYM horyzoncie na raz -- patrz pełne
# uzasadnienie projektowe w engine/sobol_analysis.py.
# ---------------------------------------------------------------------------

@app.callback(
    Output("sobol-rf-hint", "children"),
    Input("dropdown-sobol-snapshot", "value"),
)
def show_sobol_rf_hint(snapshot_id):
    """
    Confirmed 2026-09-28 (Etap 8h): pokazuje rzeczywistą rentowność 10Y
    (^TNX) na dzień utworzenia wybranego zapisu jako CZYSTY KONTEKST --
    celowo NIE zmienia automatycznie `sobol-range-rf-min`/`-max` poniżej.
    Zakres przeszukiwania rf w Sobolu zostaje świadomie szeroki i płaski
    (DEFAULT_PARAM_RANGES, Etap 8g) -- to osobna decyzja metodologiczna,
    którą właściciel projektu może zmienić ręcznie w gridzie, patrząc na tę
    podpowiedź, ale nie robimy tego za niego cicho.
    """
    if not snapshot_id:
        return ""
    record = snap.get_snapshot(snapshot_id)
    if not record:
        return ""
    created_at_str = (record.get("created_at") or "")[:10]
    if not created_at_str:
        return ""
    yield_frac = fetch_treasury_yield_on_date(created_at_str)
    if yield_frac is None:
        return f"(nie udało się pobrać ^TNX na dzień {created_at_str})"
    return f"Rentowność 10Y (^TNX) na dzień utworzenia zapisu: {yield_frac*100:.2f}% -- kontekst, nie zmienia zakresu rf poniżej."


@app.callback(
    Output("sobol-horizon-availability-note", "children"),
    Input("dropdown-sobol-snapshot", "value"), Input("dropdown-sobol-horizon", "value"),
)
def check_sobol_horizon_availability(snapshot_id, horizon_label):
    """Sprawdza, czy od utworzenia zapisu minelo wystarczajaco duzo REALNYCH
    sesji gieldowych, zeby w ogole mozna bylo ocenic wybrany horyzont bez
    look-ahead bias -- potwierdzone kluczowe zabezpieczenie (2026-09-19):
    pierwszy zapis w tym projekcie powstal 2026-09-01, wiec np. horyzont 6m
    (126 sesji) fizycznie nie moze byc jeszcze oceniony na REALNYCH danych.

    "since_inception" (dodane 2026-09-19, na prosbe wlasciciela projektu --
    pierwszy zapis mial wtedy dopiero 14 sesji, za malo na jakikolwiek staly
    horyzont): uzywa WSZYSTKICH dostepnych sesji od utworzenia zapisu do
    dzis, zamiast stalej liczby dni -- zawsze "dostepne", jesli minela
    chociaz garstka sesji, kosztem tego, ze wynik na bardzo krotkim oknie
    bedzie statystycznie szumny (ostrzezenie ponizej, nie blokada)."""
    if not snapshot_id or not horizon_label:
        return ""
    record = snap.get_snapshot(snapshot_id)
    if not record:
        return ""
    try:
        created_at = pd.Timestamp((record.get("created_at") or "")[:10])
    except (ValueError, TypeError):
        return ""
    available_days = int(np.busday_count(created_at.date(), pd.Timestamp.now().date()))

    if horizon_label == "since_inception":
        if available_days < 1:
            return "⚠ Zapis został utworzony dzisiaj -- brak jeszcze żadnej pełnej sesji do oceny."
        if available_days < 5:
            return (f"⚠ Dostępne tylko {available_days} sesji od utworzenia zapisu -- można uruchomić, ale CAGR/Sortino "
                     f"na tak krótkim oknie będą bardzo szumne (roczone z garstki dni). Traktuj wynik jako test mechanizmu, nie diagnozę.")
        return f"✓ Dostępnych {available_days} sesji od utworzenia zapisu -- \"Od początku\" użyje wszystkich {available_days}."

    needed_days = HORIZON_DAYS.get(horizon_label, 21)
    if available_days < needed_days:
        return (f"⚠ Od utworzenia tego zapisu minęło {available_days} sesji giełdowych, a horyzont {horizon_label} "
                f"potrzebuje {needed_days} REALNYCH sesji (bez zaglądania w przyszłość). Analiza nie może się jeszcze "
                f"uruchomić dla tej kombinacji zapis/horyzont -- wróć za {needed_days - available_days} sesji, albo wybierz krótszy horyzont "
                f"(albo \"Od początku\", żeby użyć tego, co już jest dostępne).")
    return f"✓ Dostępnych {available_days} sesji od utworzenia zapisu -- horyzont {horizon_label} ({needed_days} sesji) może zostać oceniony."


@app.callback(
    Output("sobol-cost-estimate", "children"),
    Input("input-sobol-n", "value"), Input("checkbox-sobol-morris-first", "value"),
)
def update_sobol_cost_estimate(n_value, morris_checked):
    n_value = int(n_value) if isinstance(n_value, (int, float)) and n_value >= 4 else 256
    cost = estimate_run_cost(n_value)
    msg = (f"Sobol: N={n_value} → {cost['n_runs']} przebiegów · na {cost['n_workers']} rdzeniach: "
           f"~{cost['est_seconds_parallel']:.0f}s (sekwencyjnie: ~{cost['est_seconds_sequential']/60:.1f} min)")
    if "ON" in (morris_checked or []):
        morris_runs = 20 * (len(PARAM_ORDER) + 1)
        msg += f"  ·  Morris (przesiew, 20 trajektorii): +{morris_runs} przebiegów, ~{morris_runs*0.05/cost['n_workers']:.0f}s"
    return msg


def _build_sobol_bar_figure(names, s1, s1_conf, st, st_conf, title):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=s1, name="S1 (pierwszego rzędu)", marker_color=THEME["accent"],
                          error_y=dict(type="data", array=s1_conf, visible=True)))
    fig.add_trace(go.Bar(x=names, y=st, name="ST (całkowity, z interakcjami)", marker_color=THEME["orange"],
                          error_y=dict(type="data", array=st_conf, visible=True)))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"],
        title=dict(text=title, font=dict(size=13, color=THEME["text_white"])),
        barmode="group", height=320, margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(tickfont=dict(size=10, color=THEME["text_dim"])),
        yaxis=dict(title="Indeks Sobola", tickfont=dict(size=10, color=THEME["text_dim"])),
        legend=dict(font=dict(size=10, color=THEME["text_white"])),
    )
    return fig


# Confirmed 2026-09-28 (Etap 8i): last Sobol batch's per-run raw data (sample +
# outputs), cached server-side so the parameter-stability panel below can be
# redrawn instantly when the top-% highlight threshold changes, WITHOUT
# re-running the (expensive, N*18-solver-call) Sobol batch itself. Deliberately
# server-side, not a dcc.Store -- run_sobol_batch's own docstring
# (engine/sobol_analysis.py) is explicit that raw_sample/raw_outputs must NOT
# be shipped to the browser (thousands of rows, too much for a JSON round-trip).
#
# BUGFIX 2026-09-28 (still Etap 8i, same-day fix): this was originally a plain
# module-level dict (`_LAST_SOBOL_RAW = {...}`), which is wrong for a
# `background=True` callback -- run_sobol_analysis below runs in a SEPARATE
# worker process spawned by DiskcacheManager (see ui/app_instance.py), which
# has its own copy of this module's globals. Writing to a module dict there
# never reached the main process, so the highlight-%-change callback (which
# DOES run in the main process, it's a normal foreground callback) always
# read back the untouched {"sample": None, ...} and silently no-op'd via
# `dash.no_update` -- the radio buttons/custom field visibly did nothing.
# Fixed by reusing the project's existing `cache` (the same diskcache.Cache
# instance DiskcacheManager itself uses, imported from ui.app_instance), which
# is an actual on-disk store both processes read/write through -- not a
# per-process Python object.
_SOBOL_STABILITY_CACHE_KEY = "sobol_stability_raw_v1"

STABILITY_PCT_OPTIONS = [10, 7.5, 5, 2.5]

# Confirmed 2026-09-30 (Etap 8m): domyślne pokrycie jądra gęstości (HDR, patrz
# _hdr_regions poniżej) -- user explicitly asked for a slider to adjust this,
# not a hardcoded constant, so DENSITY_Q_DEFAULT is only the slider's starting
# position.
DENSITY_Q_DEFAULT = 80
DENSITY_Q_MIN, DENSITY_Q_MAX, DENSITY_Q_STEP = 50, 95, 5

# Fixed marker color for bottom-N% (worst) points in the stability scatter --
# deliberately NOT one of THEME's semantic pos/neg/warn colors (those are
# reserved for the top-N% verdict), so "worst" always reads as its own,
# consistent third color regardless of that parameter's stability verdict.
BOTTOM_MARKER_COLOR = CHART_COLORS[1]  # "#B983FF", muted purple


def _stability_controls(preset_id, custom_id, label="Podświetl top:"):
    """Preset-% radio + free-text custom-% field, shared layout for the
    Sortino and CAGR stability panels below (same-day CAGR twin, 2026-09-28).
    `label` added 2026-09-30 so the same builder serves both the top-N% and
    the new, independent bottom-N% ("najgorsze") control -- same widget,
    different wording and different component ids at the call site."""
    return html.Div([
        html.Span(label, style={"fontSize": "10px", "color": THEME["text_label"], "marginRight": "8px"}),
        dcc.RadioItems(
            id=preset_id,
            options=[{"label": f" {p:g}%  ", "value": p} for p in STABILITY_PCT_OPTIONS],
            value=STABILITY_PCT_OPTIONS[0], inline=True,
            labelStyle={"fontSize": "11px", "color": THEME["text_dim"], "marginRight": "10px"},
            style={"display": "inline-block", "verticalAlign": "middle"},
        ),
        html.Span("lub własna:", style={"fontSize": "10px", "color": THEME["text_label"], "marginLeft": "12px", "marginRight": "6px"}),
        dcc.Input(
            id=custom_id, type="number", min=0.1, max=100, step=0.1,
            placeholder="np. 15", debounce=True,
            style={"width": "70px", "backgroundColor": THEME["bg_input"], "color": THEME["text_white"],
                   "border": f"1px solid {THEME['border_strong']}", "borderRadius": "4px",
                   "fontSize": "11px", "padding": "3px 6px", "verticalAlign": "middle"},
        ),
        html.Span(" %", style={"fontSize": "10px", "color": THEME["text_dim"], "marginLeft": "3px"}),
    ], style={"marginBottom": "10px"})


def _density_slider_control(slider_id):
    """New control (Etap 8m, 2026-09-30): coverage `q` of the density-based
    HDR (_hdr_regions) that now drives the stability verdict instead of a
    raw min-max span. Explicit user request: "domyślnie 80% ale daj też
    suwak" -- 80% start, freely adjustable 50-95% (below 50% the region
    becomes too small to call a "stable band" in any useful sense; above
    95% it starts re-admitting the single-outlier problem this whole
    change exists to fix)."""
    return html.Div([
        html.Span("Jądro gęstości obejmuje:", style={"fontSize": "10px", "color": THEME["text_label"], "marginRight": "10px"}),
        html.Div(
            dcc.Slider(
                id=slider_id, min=DENSITY_Q_MIN, max=DENSITY_Q_MAX, step=DENSITY_Q_STEP, value=DENSITY_Q_DEFAULT,
                marks={v: f"{v}%" for v in range(DENSITY_Q_MIN, DENSITY_Q_MAX + 1, 10)},
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            style={"width": "230px", "display": "inline-block", "verticalAlign": "middle"},
        ),
    ], style={"marginBottom": "10px", "display": "flex", "alignItems": "center"})


def _stability_controls_group(top_preset_id, top_custom_id, bottom_preset_id, bottom_custom_id, q_slider_id):
    """Groups the three independent controls one stability panel now needs:
    top-% (existing), bottom-% (new, independent per explicit request -- NOT
    tied to the top-% value), and the density-coverage slider (new). Kept as
    three separate widgets rather than one combined control, exactly per the
    user's own three answers (80%-default-plus-slider / separate bottom
    control / third color) -- this function only lays them out together."""
    return html.Div([
        _stability_controls(top_preset_id, top_custom_id, label="Podświetl top:"),
        _stability_controls(bottom_preset_id, bottom_custom_id, label="Porównaj z dołem (najgorsze):"),
        _density_slider_control(q_slider_id),
    ], style={"display": "flex", "flexWrap": "wrap", "gap": "28px", "alignItems": "flex-start", "marginBottom": "4px"})


def _hdr_regions(values, lo, hi, q, grid_n=400):
    """
    Highest-Density Region (HDR) at coverage `q` (0-100) for a 1D sample of
    parameter values, restricted to the parameter's configured slider range
    [lo, hi] -- confirmed 2026-09-30 (Etap 8m), replaces the old raw min-max
    span as the stability panels' core stability measure, and doubles as the
    input to the new two-sided/"kierunek" check in _classify_direction below.

    This is the standard "highest density interval/region" construction from
    Bayesian statistics (same idea `arviz.hdi` or a credible-region plot
    uses): fit a 1D Gaussian KDE to the sample (scipy.stats.gaussian_kde,
    Scott's rule bandwidth -- no manual tuning needed), evaluate it on a
    fine grid across [lo, hi], then threshold DOWNWARD from the highest
    density value, accumulating grid cells (by density, highest first)
    until their cumulative probability mass reaches `q`% of the total mass
    over [lo, hi] (mass the KDE puts outside [lo, hi] -- e.g. near a
    boundary -- is simply not part of this grid and is implicitly excluded,
    since only the configured slider range is a valid parameter value here).

    Thresholding a density surface (instead of taking a single min-max
    interval of raw points) can naturally return MULTIPLE disjoint regions
    when the density has more than one peak. That one property is what
    answers BOTH of the user's asks with a single computation:
      - "measure density, don't let one outlier decide the verdict": a lone
        outlier contributes only its own, low, single-kernel density bump,
        while a genuine cluster of nearby points reinforces itself (Gaussian
        kernels overlap and stack), so the top-mass grid cells overwhelmingly
        get picked from the real cluster first -- the outlier's region is
        typically never even reached at 80% coverage.
      - "does a parameter work in two directions / give best AND worst at
        the same time": if the sample genuinely clusters in two separate
        places (each cluster tall/wide enough to carry real probability
        mass), thresholding naturally returns TWO disjoint regions -- no
        separate ad hoc "look for a gap" heuristic needed, it falls out of
        the same HDR computation used for the stability number.

    Returns (regions, total_width_frac): `regions` is a list of
    (region_lo, region_hi) tuples in ascending order (grid-resolution
    accurate; always non-empty for n>=1 finite values), `total_width_frac`
    is the combined width of all regions divided by (hi-lo) -- this REPLACES
    the old `span_pct` in the caller. Falls back to a plain (min, max)
    single region -- i.e. the OLD behavior -- when there are fewer than 4
    finite values (not enough to fit a meaningful KDE) or they're all
    (near-)identical; never raises.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    span = (hi - lo) if hi > lo else 1.0
    n = len(values)
    if n == 0:
        return [], 0.0
    if n < 4 or (values.max() - values.min()) < 1e-12:
        v_lo, v_hi = float(values.min()), float(values.max())
        return [(v_lo, v_hi)], (v_hi - v_lo) / span

    try:
        kde = gaussian_kde(values)
        grid = np.linspace(lo, hi, grid_n)
        density = kde(grid)
    except Exception:
        v_lo, v_hi = float(values.min()), float(values.max())
        return [(v_lo, v_hi)], (v_hi - v_lo) / span

    dx = (hi - lo) / (grid_n - 1) if grid_n > 1 else span
    total_mass = float(density.sum() * dx)
    if total_mass <= 0 or not np.isfinite(total_mass):
        v_lo, v_hi = float(values.min()), float(values.max())
        return [(v_lo, v_hi)], (v_hi - v_lo) / span

    target_mass = (q / 100.0) * total_mass
    order = np.argsort(density)[::-1]  # highest density first
    included = np.zeros(grid_n, dtype=bool)
    cum = 0.0
    for idx in order:
        included[idx] = True
        cum += density[idx] * dx
        if cum >= target_mass:
            break

    regions = []
    in_region, start = False, None
    for i in range(grid_n):
        if included[i] and not in_region:
            in_region, start = True, grid[i]
        elif not included[i] and in_region:
            in_region = False
            regions.append((start, grid[i - 1]))
    if in_region:
        regions.append((start, grid[-1]))
    if not regions:  # defensive -- shouldn't happen once target_mass>0, but never raise
        v_lo, v_hi = float(values.min()), float(values.max())
        return [(v_lo, v_hi)], (v_hi - v_lo) / span

    total_width = sum(b - a for a, b in regions)
    return regions, total_width / span


def _classify_direction(top_regions, bottom_regions, lo, hi):
    """
    "Kierunek" verdict (Etap 8m, 2026-09-30) -- the user's point 1 ask:
    "czy może parametr nie działa w dwie strony i może jednocześnie dać
    najlepsze jak i najgorsze wartości". Built entirely on the HDR regions
    _hdr_regions already computed for top-N% and bottom-N% -- no separate
    model fit.

    Three outcomes:
      - "BEZ WYRAŹNEGO WPŁYWU": top-N%'s and bottom-N%'s HDRs overlap by
        more than half of the top region's own width -- best and worst runs
        come from the same place on this parameter's axis, so it isn't
        discriminating outcome quality here (this can fire even when the
        top-N% band itself is narrow -- a narrow band that ALSO produces
        the worst outcomes is exactly "this parameter doesn't matter", not
        "stable", which is why this check runs before the stability verdict
        would otherwise get the last word).
      - "DWUKIERUNKOWY": top-N%'s HDR itself is multi-region (after merging
        regions closer together than 2% of the axis -- grid noise, not a
        real second band) -- i.e. good runs cluster in more than one place.
        When bottom-N%'s HDR sits specifically IN THE GAP between the top
        regions, that's a strong, explicit confirmation (classic U-shape:
        both extremes good, middle bad); otherwise it's still reported, with
        a softer note pointing at a likely interaction with another
        parameter as the more probable explanation.
      - "JEDNOKIERUNKOWY": top-N% is a single region and doesn't overlap
        much with bottom-N% -- the ordinary, single-direction case.

    Returns (label, color, note) -- `note` is "" when there's nothing extra
    to say (e.g. no bottom-N% sample was available to compare against).
    """
    if not top_regions:
        return "BRAK DANYCH", THEME["text_dim"], ""

    def _width(regions):
        return sum(b - a for a, b in regions)

    def _overlap(regions_a, regions_b):
        total = 0.0
        for a0, a1 in regions_a:
            for b0, b1 in regions_b:
                total += max(0.0, min(a1, b1) - max(a0, b0))
        return total

    top_w = _width(top_regions)
    overlap_ratio = (_overlap(top_regions, bottom_regions) / top_w) if (bottom_regions and top_w > 0) else 0.0

    if bottom_regions and overlap_ratio > 0.5:
        return "BEZ WYRAŹNEGO WPŁYWU", THEME["text_dim"], "najlepsze i najgorsze przebiegi trafiają w te same wartości tego parametru"

    span = hi - lo
    min_region_width = 0.01 * span
    merge_gap = 0.02 * span
    sig_regions = sorted((a, b) for a, b in top_regions if (b - a) >= min_region_width) or sorted(top_regions)
    merged = []
    for a, b in sig_regions:
        if merged and a - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))

    if len(merged) >= 2:
        gaps = [(merged[i][1], merged[i + 1][0]) for i in range(len(merged) - 1)]
        note = "brak dołu do porównania"
        if bottom_regions:
            bottom_mid = sum((a + b) / 2.0 for a, b in bottom_regions) / len(bottom_regions)
            if any(g0 <= bottom_mid <= g1 for g0, g1 in gaps):
                note = "najgorsze przebiegi skupiają się właśnie pomiędzy tymi pasmami -- silne potwierdzenie"
            else:
                note = "dół nie leży w przerwie -- sprawdź, czy o wyborze pasma decyduje inny parametr (interakcja)"
        return "DWUKIERUNKOWY", THEME["warn"], note

    if bottom_regions:
        return "JEDNOKIERUNKOWY", THEME["pos"], "najlepsze i najgorsze przebiegi trzymają się osobnych pasm"
    return "JEDNOKIERUNKOWY", THEME["pos"], ""


def _build_stability_panel(sample, raw_outputs, param_ranges, pct_top, pct_bottom, q, metric="Sortino"):
    """
    Confirmed 2026-09-28 (Etap 8i); REDESIGNED 2026-09-30 (Etap 8m) to fix an
    outlier-fragility problem the owner flagged: the original verdict was the
    raw min-max SPAN of the top-`pct_top`% points on a parameter's axis, so a
    single point that landed in the top-N% by pure chance (e.g. through a
    strong interaction with another parameter) but sits at an extreme value
    could single-handedly flip a genuinely tight cluster's verdict from
    STABILNE to NIESTABILNE. Also added, same request: a second, independent
    bottom-`pct_bottom`% ("najgorsze") sample, to check whether a parameter
    "works in two directions" -- clusters BOTH the best and the worst runs,
    just in different places, rather than simply not mattering.

    Both asks are now answered by ONE shared computation: `_hdr_regions`
    (Highest-Density Region at coverage `q`, via a Gaussian KDE -- see its
    own docstring for the full rationale) run once on the top-N% sample and
    once on the bottom-N% sample, for each parameter:
      - The stability verdict below now comes from the top-N% HDR's total
        width (as % of the full slider range), NOT the raw min-max -- a lone
        outlier typically carries too little density to be pulled into the
        HDR at all, so it no longer drives the verdict. The raw min-max is
        still shown, as a small non-alarming footnote, whenever it differs
        materially from the HDR (so outliers stay visible, just not decisive).
      - The new "kierunek" line (_classify_direction) compares the top-N% and
        bottom-N% HDRs: heavy overlap says the parameter isn't discriminating
        outcome quality here at all; a top-N% HDR that itself splits into two
        separate bands says the parameter has more than one good regime
        ("DWUKIERUNKOWY"), especially convincing when the bottom-N% HDR sits
        specifically in the gap between them.

    `metric`: "Sortino" (default) or "CAGR" -- the two are independent panels
    in results_layout below (own controls each), not a toggle on one panel,
    since CAGR/Sortino can disagree on which runs are "best"/"worst".

    This still shows CORRELATION, not causation -- with all 8 parameters
    varying together per Saltelli sample, an apparent cluster could be driven
    by an interaction with another parameter rather than this one alone.
    That's exactly what the existing S1 (own effect) / ST (own effect +
    interactions) bar charts and PRCC table already measure formally; this
    panel (now plus its "kierunek" line) is a visual complement to them, not
    a replacement -- read together.

    Rows with solver_success=False are excluded ENTIRELY (not median-imputed
    the way the Sobol variance-decomposition math itself handles them) --
    injecting a fake median point here would visually distort where winning
    (and losing) runs actually cluster.
    """
    solver_success = raw_outputs["solver_success"].values
    metric_all = raw_outputs[metric].values
    valid = solver_success & np.isfinite(metric_all)
    n_valid = int(valid.sum())
    if n_valid < 20:
        return html.Div(
            f"Za mało poprawnych przebiegów ({n_valid}) do sensownego panelu stabilności -- "
            f"potrzeba co najmniej 20 (po odrzuceniu awarii solvera).",
            style={"fontSize": "11px", "color": THEME["warn"]}
        )

    metric_valid = metric_all[valid]
    sample_valid = np.asarray(sample)[valid]

    n_top = max(1, round(pct_top / 100.0 * n_valid))
    n_bottom = max(1, round(pct_bottom / 100.0 * n_valid)) if pct_bottom and pct_bottom > 0 else 0
    if n_top + n_bottom > n_valid:
        # Top and bottom are requested independently (explicit ask) -- if
        # their combined size would exceed the sample (e.g. both set to a
        # large %), scale both down proportionally rather than letting them
        # silently overlap (a run can't be simultaneously "best" and "worst").
        scale = n_valid / float(n_top + n_bottom)
        n_top = max(1, int(round(n_top * scale)))
        n_bottom = max(0, min(n_valid - n_top, int(round(n_bottom * scale))))

    order = np.argsort(metric_valid)  # ascending by metric
    top_idx = order[-n_top:]
    bottom_idx = order[:n_bottom] if n_bottom > 0 else np.array([], dtype=int)
    excluded = set(top_idx.tolist()) | set(bottom_idx.tolist())
    rest_idx = np.array([i for i in range(n_valid) if i not in excluded], dtype=int)

    q = q if isinstance(q, (int, float)) and 0 < q <= 100 else DENSITY_Q_DEFAULT

    panels = []
    for i, pname in enumerate(PARAM_ORDER):
        col = sample_valid[:, i]
        lo, hi = param_ranges.get(pname, DEFAULT_PARAM_RANGES[pname])
        full_span = (hi - lo) if hi > lo else 1.0

        top_vals = col[top_idx]
        bottom_vals = col[bottom_idx] if n_bottom > 0 else np.array([])

        top_regions, top_w_frac = _hdr_regions(top_vals, lo, hi, q)
        bottom_regions, _ = _hdr_regions(bottom_vals, lo, hi, q) if n_bottom > 0 else ([], 0.0)

        span_pct = 100.0 * top_w_frac
        if span_pct < 25:
            verdict, color = "STABILNE", THEME["pos"]
        elif span_pct < 60:
            verdict, color = "UMIARKOWANE", THEME["warn"]
        else:
            verdict, color = "NIESTABILNE", THEME["neg"]

        direction_label, direction_color, direction_note = _classify_direction(top_regions, bottom_regions, lo, hi)

        # Raw (non-robust) min-max of top-N%, kept ONLY as an outlier-visibility
        # footnote -- this is the old metric, and it must never again drive the
        # verdict above (that's the exact bug this redesign fixes).
        raw_lo, raw_hi = float(top_vals.min()), float(top_vals.max())
        raw_span_pct = 100.0 * (raw_hi - raw_lo) / full_span
        n_outside_core = int(sum(1 for v in top_vals if not any(a <= v <= b for a, b in top_regions)))
        outlier_note = ""
        if raw_span_pct - span_pct > 15 and n_outside_core > 0:
            outlier_note = f" (surowy zakres: {raw_span_pct:.0f}%, {n_outside_core} pkt poza jądrem)"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=col[rest_idx], y=metric_valid[rest_idx], mode="markers",
            marker=dict(size=4, color=THEME["text_dim"], opacity=0.30),
            showlegend=False, hoverinfo="skip"))
        if n_bottom > 0:
            fig.add_trace(go.Scatter(
                x=bottom_vals, y=metric_valid[bottom_idx], mode="markers",
                marker=dict(size=5, color=BOTTOM_MARKER_COLOR, opacity=0.85), showlegend=False,
                hovertemplate=f"{pname}=%{{x:.3f}}<br>{metric}=%{{y:.3f}} (najgorsze)<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=top_vals, y=metric_valid[top_idx], mode="markers",
            marker=dict(size=5, color=color, opacity=0.9), showlegend=False,
            hovertemplate=f"{pname}=%{{x:.3f}}<br>{metric}=%{{y:.3f}} (najlepsze)<extra></extra>"))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"],
            height=225, margin=dict(l=36, r=10, t=28, b=28),
            title=dict(text=pname, font=dict(size=11, color=THEME["text_white"])),
            xaxis=dict(tickfont=dict(size=9, color=THEME["text_dim"])),
            yaxis=dict(title=metric, title_font=dict(size=9, color=THEME["text_dim"]),
                       tickfont=dict(size=9, color=THEME["text_dim"])),
        )

        panels.append(html.Div([
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
            html.Div(f"gęstość top {pct_top:g}% (jądro {q:g}%): {span_pct:.0f}% zakresu -- {verdict}{outlier_note}",
                     style={"fontSize": "9.5px", "color": color, "textAlign": "center", "marginTop": "2px"}),
            html.Div(f"kierunek: {direction_label}" + (f" -- {direction_note}" if direction_note else ""),
                     style={"fontSize": "9px", "color": direction_color, "textAlign": "center", "marginTop": "1px", "fontStyle": "italic"}),
        ]))

    return html.Div([
        html.Div(
            f"({n_valid} poprawnych przebiegów z {len(raw_outputs)}; top {pct_top:g}% = {n_top} pkt (zielone/żółte/czerwone, wg werdyktu), "
            f"dół {pct_bottom:g}% = {n_bottom} pkt (fioletowe), jądro gęstości = {q:g}%, wg {metric})",
            style={"fontSize": "10px", "color": THEME["text_dim"], "marginBottom": "10px"}),
        html.Div(panels, style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "10px"}),
    ])


def _build_best_combo_readout(sample, raw_outputs):
    """
    Same-day addition (2026-09-28): "daj parametry dla kombinacji parametrów,
    która miała najwyższy CAGR oraz Sortino Ratio". The single Saltelli run
    (out of the whole batch) that scored highest on each metric, with its
    exact 8 parameter values read off straight from raw_sample/raw_outputs --
    zero extra solver calls, same cached data as the stability panels above.

    Deliberately NOT presented as "recommended settings": this is one sampled
    point in an 8-dimensional space, not a re-optimization or a centroid of
    the stable region above -- with N runs it can land on a lucky outlier,
    especially in thin, noisy corners of the search range. Read it alongside
    the stability panels (a parameter that's also STABILE there gives this
    point more weight than one that's NIESTABILNE).
    """
    solver_success = raw_outputs["solver_success"].values
    cagr_all = raw_outputs["CAGR"].values
    sortino_all = raw_outputs["Sortino"].values
    valid = solver_success & np.isfinite(cagr_all) & np.isfinite(sortino_all)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return html.Div("Brak poprawnych przebiegów do wyznaczenia najlepszej kombinacji.",
                         style={"fontSize": "11px", "color": THEME["warn"]})

    sample_valid = np.asarray(sample)[valid]
    cagr_valid = cagr_all[valid]
    sortino_valid = sortino_all[valid]

    idx_cagr = int(np.argmax(cagr_valid))
    idx_sortino = int(np.argmax(sortino_valid))

    def _row(title, idx, color):
        params_txt = ", ".join(f"{p}={sample_valid[idx, i]:.3g}" for i, p in enumerate(PARAM_ORDER))
        return html.Div([
            html.Span(title, style={"fontSize": "10px", "fontWeight": "bold", "color": color, "marginRight": "6px"}),
            html.Span(f"CAGR={cagr_valid[idx]:.2%}  Sortino={sortino_valid[idx]:.3f}  —  {params_txt}",
                      style={"fontSize": "10px", "color": THEME["text_dim"]}),
        ], style={"marginBottom": "5px"})

    return html.Div([
        _row("Najlepszy CAGR:", idx_cagr, THEME["pos"]),
        _row("Najlepszy Sortino:", idx_sortino, THEME["accent"]),
        html.Div(f"({n_valid} poprawnych przebiegów przeszukanych)",
                 style={"fontSize": "9px", "color": THEME["text_dim"], "marginTop": "2px"}),
        html.Div("Pojedynczy najlepszy PRZEBIEG próbkowania Saltelli, nie ponowna optymalizacja ani centroid stabilnego regionu z paneli powyżej -- traktuj jako punkt odniesienia, nie \"zalecane ustawienia\"; parametr STABILNY w panelu powyżej nadaje tej wartości więcej wagi niż NIESTABILNY.",
                 style={"fontSize": "9px", "color": THEME["text_dim"], "fontStyle": "italic", "marginTop": "4px"}),
    ])


def _valid_custom_pct(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v) or v <= 0:
        return None
    return min(v, 100.0)


def _resolve_stability_pct(trigger_id, custom_input_id, preset_pct, custom_pct):
    """
    Shared by both the Sortino and CAGR stability-panel rebuild callbacks
    below. Whichever control the user actually just touched wins -- via
    callback_context, same trigger_id pattern as delete_snapshot_guarded
    elsewhere in this file -- so typing a custom % doesn't fight with a
    still-selected preset radio, and picking a preset cleanly overrides a
    stale custom value left in its box.
    """
    if trigger_id == custom_input_id:
        pct = _valid_custom_pct(custom_pct)
        return pct if pct is not None else (preset_pct if isinstance(preset_pct, (int, float)) else STABILITY_PCT_OPTIONS[0])
    if isinstance(preset_pct, (int, float)):
        return preset_pct
    pct = _valid_custom_pct(custom_pct)
    return pct if pct is not None else STABILITY_PCT_OPTIONS[0]


@app.callback(
    Output("sobol-stability-panel", "children"),
    Input("sobol-stability-pct", "value"),
    Input("sobol-stability-custom-pct", "value"),
    Input("sobol-stability-pct-bottom", "value"),
    Input("sobol-stability-custom-pct-bottom", "value"),
    Input("sobol-density-q", "value"),
    prevent_initial_call=True,
)
def rebuild_stability_panel(preset_pct, custom_pct, preset_pct_bottom, custom_pct_bottom, q):
    """Cheap redraw only -- reuses the last batch's raw data from the shared
    diskcache (see _SOBOL_STABILITY_CACHE_KEY), never re-runs the solver.
    Extended 2026-09-30 (Etap 8m) with the independent bottom-% control and
    the density-coverage slider -- both resolved the same way the existing
    top-% control already was (whichever control the user just touched wins,
    via _resolve_stability_pct/callback_context, called once per pct pair)."""
    cached = cache.get(_SOBOL_STABILITY_CACHE_KEY)
    if not cached:
        return dash.no_update
    trigger_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0] if dash.callback_context.triggered else None
    pct_top = _resolve_stability_pct(trigger_id, "sobol-stability-custom-pct", preset_pct, custom_pct)
    pct_bottom = _resolve_stability_pct(trigger_id, "sobol-stability-custom-pct-bottom", preset_pct_bottom, custom_pct_bottom)
    q = q if isinstance(q, (int, float)) and 0 < q <= 100 else DENSITY_Q_DEFAULT
    return _build_stability_panel(cached["sample"], cached["raw_outputs"], cached["param_ranges"], pct_top, pct_bottom, q, metric="Sortino")


@app.callback(
    Output("sobol-stability-panel-cagr", "children"),
    Input("sobol-stability-pct-cagr", "value"),
    Input("sobol-stability-custom-pct-cagr", "value"),
    Input("sobol-stability-pct-bottom-cagr", "value"),
    Input("sobol-stability-custom-pct-bottom-cagr", "value"),
    Input("sobol-density-q-cagr", "value"),
    prevent_initial_call=True,
)
def rebuild_stability_panel_cagr(preset_pct, custom_pct, preset_pct_bottom, custom_pct_bottom, q):
    """
    Same-day addition (2026-09-28): "te same wykresy ale dla CAGR" -- an
    independent CAGR-ranked twin of rebuild_stability_panel above, own
    highlight-% controls, same cached raw data, zero extra solver calls.
    A run can rank very differently by CAGR vs. by Sortino (CAGR ignores
    downside shape entirely), so this is a second panel, not a toggle on
    the first one. Extended 2026-09-30 (Etap 8m) the same way as its Sortino
    twin -- own independent bottom-% control and density slider.
    """
    cached = cache.get(_SOBOL_STABILITY_CACHE_KEY)
    if not cached:
        return dash.no_update
    trigger_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0] if dash.callback_context.triggered else None
    pct_top = _resolve_stability_pct(trigger_id, "sobol-stability-custom-pct-cagr", preset_pct, custom_pct)
    pct_bottom = _resolve_stability_pct(trigger_id, "sobol-stability-custom-pct-bottom-cagr", preset_pct_bottom, custom_pct_bottom)
    q = q if isinstance(q, (int, float)) and 0 < q <= 100 else DENSITY_Q_DEFAULT
    return _build_stability_panel(cached["sample"], cached["raw_outputs"], cached["param_ranges"], pct_top, pct_bottom, q, metric="CAGR")


@app.callback(
    Output("sobol-results-container", "children"), Output("sobol-run-status", "children"),
    Input("btn-run-sobol", "n_clicks"),
    State("dropdown-sobol-snapshot", "value"), State("dropdown-sobol-horizon", "value"),
    State("input-sobol-n", "value"), State("checkbox-sobol-morris-first", "value"),
    *[State(f"sobol-range-{p}-min", "value") for p in PARAM_ORDER],
    *[State(f"sobol-range-{p}-max", "value") for p in PARAM_ORDER],
    background=True, progress=[Output("sobol-run-status", "children", allow_duplicate=True)],
    prevent_initial_call=True,
)
def run_sobol_analysis(set_progress, _n_clicks, snapshot_id, horizon_label, n_value, morris_checked, *range_values):
    """
    Confirmed pipeline (2026-09-19): pobiera JEDEN zapis, zamraża jego ceny
    na dwie CZĘŚCI -- risk_prices (do i wliczajac dzien utworzenia, do
    estymacji Sigma_eps/K -- ten sam ~5-letni konwencja co reszta Sandboxa)
    i forward_returns (REALNE, juz zaszle sesje PO dniu utworzenia,
    obciete dokladnie do wybranego horyzontu -- zero danych z przyszlosci
    wzgledem TEGO zapisu mozliwe z konstrukcji, bo "przyszlosc" tej analizy
    to i tak juz przeszlosc wzgledem dzisiejszej daty).
    """
    theta_mins = range_values[:len(PARAM_ORDER)]
    theta_maxs = range_values[len(PARAM_ORDER):]
    param_ranges = {p: (float(lo) if isinstance(lo, (int, float)) else DEFAULT_PARAM_RANGES[p][0],
                         float(hi) if isinstance(hi, (int, float)) else DEFAULT_PARAM_RANGES[p][1])
                    for p, lo, hi in zip(PARAM_ORDER, theta_mins, theta_maxs)}

    if not snapshot_id:
        return html.Div(), html.Div("Wybierz zapisany portfel z listy powyżej.", style={"color": THEME["orange"]})
    record = snap.get_snapshot(snapshot_id)
    if not record:
        return html.Div(), html.Div("Nie znaleziono wybranego zapisu.", style={"color": THEME["orange"]})

    try:
        created_at = pd.Timestamp((record.get("created_at") or "")[:10])
    except (ValueError, TypeError):
        return html.Div(), html.Div("Zapis nie ma poprawnej daty utworzenia.", style={"color": THEME["orange"]})

    available_days = int(np.busday_count(created_at.date(), pd.Timestamp.now().date()))
    if horizon_label == "since_inception":
        horizon_days = available_days
        if horizon_days < 3:
            return html.Div(), html.Div(
                f"Za mało sesji od utworzenia zapisu ({horizon_days}) -- potrzeba co najmniej 3, żeby policzyć cokolwiek sensownego.",
                style={"color": THEME["orange"]})
    else:
        horizon_days = HORIZON_DAYS.get(horizon_label, 21)
        if available_days < horizon_days:
            return html.Div(), html.Div(
                f"Za mało realnych sesji od utworzenia zapisu ({available_days} < {horizon_days} potrzebnych dla {horizon_label}).",
                style={"color": THEME["orange"]})

    tickers = list(record.get("final_weights", {}).keys())
    if len(tickers) < 2:
        return html.Div(), html.Div("Zapis ma mniej niż 2 spółki -- za mało do analizy.", style={"color": THEME["orange"]})

    set_progress(["Pobieram historię cen (od 5 lat przed dniem zapisu do dziś)..."])
    fetch_start = (created_at - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    try:
        all_prices = snap.fetch_price_history(tickers, fetch_start)
    except Exception as e:
        return html.Div(), html.Div(f"Błąd pobierania cen: {e}", style={"color": THEME["orange"]})
    if all_prices.empty:
        return html.Div(), html.Div("Nie udało się pobrać żadnych cen.", style={"color": THEME["orange"]})

    risk_prices = all_prices[all_prices.index <= created_at].dropna(axis=1, how="all")
    forward_prices_full = all_prices[all_prices.index > created_at]
    if risk_prices.shape[1] < 2 or len(risk_prices) < 20:
        return html.Div(), html.Div("Za mało wspólnej historii cenowej sprzed dnia zapisu do estymacji ryzyka.", style={"color": THEME["orange"]})

    common_tickers = [t for t in risk_prices.columns if t in forward_prices_full.columns]
    forward_prices = forward_prices_full[common_tickers].iloc[:horizon_days]
    combined_for_returns = pd.concat([risk_prices[common_tickers].iloc[[-1]], forward_prices])
    forward_returns = np.log(combined_for_returns / combined_for_returns.shift(1)).dropna()
    if len(forward_returns) < horizon_days - 2:  # mala tolerancja na dni bez notowan (swieta itp.)
        return html.Div(), html.Div(
            f"Za mało realnych sesji cenowych w oknie forward ({len(forward_returns)} < ~{horizon_days}).",
            style={"color": THEME["orange"]})

    fundamental_inputs = record.get("fundamental_inputs", [])
    cluster_of = record.get("cluster_of", {t: 1 for t in common_tickers})
    n_value = int(n_value) if isinstance(n_value, (int, float)) and n_value >= 4 else 256

    morris_summary = None
    if "ON" in (morris_checked or []):
        set_progress(["Uruchamiam tani przesiew Morrisa (20 trajektorii)..."])
        def _morris_progress(completed, total, elapsed):
            set_progress([f"Morris: {completed}/{total} przebiegów ({elapsed:.0f}s)..."])
        morris_summary = run_morris_screen(fundamental_inputs, cluster_of, risk_prices[common_tickers], forward_returns,
                                             param_ranges=param_ranges, n_trajectories=20, on_progress=_morris_progress)

    def _sobol_progress(completed, total, elapsed):
        if completed % max(total // 100, 1) == 0 or completed == total:
            set_progress([f"Sobol: {completed}/{total} przebiegów ({elapsed:.0f}s, ~{elapsed/max(completed,1)*(total-completed):.0f}s pozostało)..."])

    set_progress([f"Uruchamiam pełną analizę Sobola (N={n_value})..."])
    sobol_result = run_sobol_batch(fundamental_inputs, cluster_of, risk_prices[common_tickers], forward_returns,
                                     param_ranges=param_ranges, N=n_value, on_progress=_sobol_progress)

    # Etap 8i: cache raw per-run data server-side for the stability panel below --
    # see _SOBOL_STABILITY_CACHE_KEY's own comment above for why this goes through
    # the shared diskcache.Cache (not a module global -- this callback runs in a
    # separate background-callback worker process).
    cache.set(_SOBOL_STABILITY_CACHE_KEY, {
        "sample": sobol_result["raw_sample"],
        "raw_outputs": sobol_result["raw_outputs"],
        "param_ranges": param_ranges,
    })

    # --- Renderowanie wynikow ---
    sections = []
    if morris_summary is not None:
        morris_rows = []
        for pname in PARAM_ORDER:
            row = {"Parametr": pname}
            for output_name in ["CAGR", "Sortino", "MaxDrawdown"]:
                out = morris_summary["outputs"].get(output_name, {})
                if "error" not in out:
                    idx = out["names"].index(pname)
                    row[f"{output_name} μ*"] = round(out["mu_star"][idx], 4)
            morris_rows.append(row)
        sections.append(html.Div([
            html.Div(f"PRZESIEW MORRISA ({morris_summary['n_runs']} przebiegów, {morris_summary['n_solver_failures']} awarii solvera)",
                     style={"fontSize": "11px", "fontWeight": "bold", "color": THEME["text_white"], "marginBottom": "8px"}),
            dash_table.DataTable(
                columns=[{"name": c, "id": c} for c in morris_rows[0].keys()] if morris_rows else [],
                data=morris_rows, style_header=datatable_style_header(), style_data=datatable_style_data(), style_cell=datatable_style_cell(),
                style_data_conditional=[datatable_row_alt_rule()],
            ),
        ], style={"marginBottom": "24px"}))

    charts = []
    for output_name in ["CAGR", "Sortino", "MaxDrawdown"]:
        out = sobol_result["sobol"].get(output_name, {})
        if "error" in out:
            charts.append(html.Div(f"{output_name}: błąd analizy Sobola -- {out['error']}", style={"color": THEME["orange"], "fontSize": "11px"}))
            continue
        charts.append(dcc.Graph(figure=_build_sobol_bar_figure(out["names"], out["S1"], out["S1_conf"], out["ST"], out["ST_conf"], output_name),
                                  config={"displayModeBar": False}))

    prcc_rows = []
    for pname in PARAM_ORDER:
        row = {"Parametr": pname}
        for output_name in ["CAGR", "Sortino", "MaxDrawdown"]:
            v = sobol_result["prcc"].get(output_name, {}).get(pname)
            row[output_name] = round(v, 4) if v is not None and np.isfinite(v) else "n/d"
        prcc_rows.append(row)

    n_fail = sobol_result["n_solver_failures"]
    fail_note = html.Div()
    if n_fail > 0:
        fail_pct = 100 * n_fail / max(sobol_result["n_runs"], 1)
        fail_note = html.Div(
            f"⚠ {n_fail}/{sobol_result['n_runs']} przebiegów ({fail_pct:.1f}%) nie zbiegło się -- zastąpione medianą próbki. "
            f"Przy dużym odsetku awarii wyniki poniżej mogą być zniekształcone.",
            style={"fontSize": "10.5px", "color": THEME["warn"], "marginBottom": "12px"}
        )

    small_n_note = html.Div(
        "Uwaga: przy małym N pojedyncze wartości S1 mogą wyjść nieznacznie ujemne mimo że S1≥0 teoretycznie -- "
        "to znany artefakt estymatora przy małej próbce (wartość bliska zeru), nie błąd.",
        style={"fontSize": "10px", "color": THEME["text_dim"], "marginBottom": "12px", "fontStyle": "italic"}
    ) if n_value < 128 else html.Div()

    results_layout = html.Div([
        *sections,
        fail_note, small_n_note,
        html.Div("INDEKSY SOBOLA (S1 = wkład sam w sobie, ST = wkład razem z interakcjami)",
                 style={"fontSize": "11px", "fontWeight": "bold", "color": THEME["text_white"], "marginBottom": "8px"}),
        html.Div(charts, style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(380px, 1fr))", "gap": "12px", "marginBottom": "20px"}),
        html.Div("PRCC (niezależny sprawdzian krzyżowy rankingu, ta sama próbka co Sobol)",
                 style={"fontSize": "11px", "fontWeight": "bold", "color": THEME["text_white"], "marginBottom": "8px"}),
        dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in prcc_rows[0].keys()] if prcc_rows else [],
            data=prcc_rows, style_header=datatable_style_header(), style_data=datatable_style_data(), style_cell=datatable_style_cell(),
            style_data_conditional=[datatable_row_alt_rule()],
        ),

        html.Div("NAJLEPSZA POJEDYNCZA KOMBINACJA PARAMETRÓW (CAGR / Sortino)",
                 style={"fontSize": "11px", "fontWeight": "bold", "color": THEME["text_white"], "marginBottom": "8px", "marginTop": "24px"}),
        html.Div(_build_best_combo_readout(sobol_result["raw_sample"], sobol_result["raw_outputs"]),
                 style={"marginBottom": "8px"}),

        html.Div("PANEL STABILNOŚCI PARAMETRÓW (Sortino) -- gdzie klastrują się najlepsze (i najgorsze) wyniki",
                 style={"fontSize": "11px", "fontWeight": "bold", "color": THEME["text_white"], "marginBottom": "8px", "marginTop": "24px"}),
        _stability_controls_group("sobol-stability-pct", "sobol-stability-custom-pct",
                                    "sobol-stability-pct-bottom", "sobol-stability-custom-pct-bottom",
                                    "sobol-density-q"),
        html.Div(id="sobol-stability-panel",
                  children=_build_stability_panel(sobol_result["raw_sample"], sobol_result["raw_outputs"],
                                                    param_ranges, STABILITY_PCT_OPTIONS[0], STABILITY_PCT_OPTIONS[0],
                                                    DENSITY_Q_DEFAULT, metric="Sortino")),

        html.Div("PANEL STABILNOŚCI PARAMETRÓW (CAGR) -- gdzie klastrują się najlepsze (i najgorsze) wyniki",
                 style={"fontSize": "11px", "fontWeight": "bold", "color": THEME["text_white"], "marginBottom": "8px", "marginTop": "24px"}),
        _stability_controls_group("sobol-stability-pct-cagr", "sobol-stability-custom-pct-cagr",
                                    "sobol-stability-pct-bottom-cagr", "sobol-stability-custom-pct-bottom-cagr",
                                    "sobol-density-q-cagr"),
        html.Div(id="sobol-stability-panel-cagr",
                  children=_build_stability_panel(sobol_result["raw_sample"], sobol_result["raw_outputs"],
                                                    param_ranges, STABILITY_PCT_OPTIONS[0], STABILITY_PCT_OPTIONS[0],
                                                    DENSITY_Q_DEFAULT, metric="CAGR")),
    ])

    horizon_display = "od początku" if horizon_label == "since_inception" else horizon_label
    status = html.Div(
        f"Gotowe -- {sobol_result['n_runs']} przebiegów Sobola" + (f" + {morris_summary['n_runs']} Morrisa" if morris_summary else "") +
        f", zapis {snapshot_id}, horyzont {horizon_display} ({horizon_days} sesji, dane wyłącznie {created_at.date()} → {forward_prices.index[-1].date() if len(forward_prices) else '?'}).",
        style={"color": THEME["accent"]}
    )
    return results_layout, status