"""
ui/tab4_rebalance.py
======================
Tab 4: Strategy Params & TPS Optimization -- the True Two-Stage SLSQP
solver (Section 3.4) + Dynamic Singleton Split, results table/donut,
and portfolio snapshot save (both from here and from Sandbox).

Confirmed removal (2026-09-18): the 3D risk-reward scatter
(render_stage4b_3d_surface, graph-stage4b-3d) was dropped from the UI --
project owner judged it added no diagnostic value over the existing
results table + donut. Removed the callback entirely rather than leaving
it orphaned against a non-existent Output.

Moved out of quant_terminal.py (Etap 0 architecture split, PROJECT_CONTEXT.md)
with NO behavior change.
"""
import dash
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State

from ui.app_instance import app
from ui.theme import THEME, CLUSTER_PALETTE
from ui.components import build_kpi_card, build_kpi_strip, STAGE4A_PARAMS_CONFIG, datatable_style_header, datatable_style_cell, datatable_style_data, datatable_row_alt_rule, build_pair_overlay_output, build_tps_formula_breakdown
from ui.tab3_tailrisk import compute_tail_penalty
from engine.risk import compute_estrada_matrix, semideviation_ann_from_matrix
from engine.optimizer import compute_asset_tps, run_stage2_inter_cluster_slsqp, run_optimization_with_singleton_split
from engine.evaluation import evaluate_portfolio_performance
from engine.returns import compute_composite_upside_row
from engine.pairs import find_matched_pairs_for_overlay, apply_tilts_to_matched_pairs
from data import snapshot_store as snap
from data.market_data import fetch_universe_prices

@app.callback(Output("panel-stage4a-container", "style"), Input("store-stage3-final-payload", "data"), prevent_initial_call=True)
def reveal_stage4a_panel(payload):
    if payload:
        return {"display": "block", "marginBottom": "35px"}
    return {"display": "none", "marginBottom": "35px"}


@app.callback(Output("panel-stage4b-container", "style"), Input("store-stage3-final-payload", "data"), prevent_initial_call=True)
def reveal_stage4b_panel(payload):
    if payload:
        return {"display": "block", "marginBottom": "35px", "marginTop": "35px"}
    return {"display": "none", "marginBottom": "35px", "marginTop": "35px"}


@app.callback(
    Output("store-stage4b-results", "data"),
    Input("input-alpha", "value"),
    Input("input-lambda", "value"), Input("input-nu", "value"), Input("input-gamma", "value"), Input("input-kappa", "value"), Input("input-nref", "value"),
    Input("input-wmax", "value"), Input("input-rf", "value"),
    Input("store-stage3-final-payload", "data"), Input("store-stage4a-tailrisk", "data"), Input("store-crash-matrices", "data"),
    State("store-daily-returns", "data"), State("store-raw-close", "data"),
    prevent_initial_call=True
)
def run_stage4b_solver(alpha, lam, nu, gamma, kappa, n_ref, w_max, rf,
                        stage3_payload, tailrisk, crash_data, daily_returns_data, raw_close_data):
    """
    Silnik Stage 4B — True Two-Stage SLSQP (Section 3.4) + Dynamic Singleton Split.
    Łączy Section 4.2 (mu_i, z Stage 3+4A), macierz Estrady Sigma_eps (Tikhonov-regularyzowaną),
    macierz Discrete Crash-Overlap K (Section 3.3) i run_optimization_with_singleton_split()
    w jeden pipeline dający finalne wagi portfela.

    Stary P_i (Z-score tail penalty) jest TU liczony WYŁĄCZNIE do celów diagnostycznych
    (Standalone Asset TPS w tabeli, wykres underwater w Tab 3) -- solver go nie dotyka,
    zgodnie z decyzją (d)/(e) z ustaleń refaktoryzacji.

    Zwraca dict {"error": "..."} zamiast rzucać wyjątkiem, żeby UI mógł pokazać czytelny komunikat
    zamiast białej strony / cichego zawieszenia callbacku.
    """
    if not stage3_payload or not tailrisk or not tailrisk.get("tickers") or not daily_returns_data or not raw_close_data or not crash_data:
        return {"error": "Brak danych — upewnij się, że Stage 1-3 zostały uruchomione i Stage 3 zatwierdzony (CONFIRM & EXPORT)."}

    alpha = alpha if isinstance(alpha, (int, float)) else 0.5
    lam = lam if isinstance(lam, (int, float)) else 3.0
    gamma = gamma if isinstance(gamma, (int, float)) else 1.50
    kappa = kappa if isinstance(kappa, (int, float)) else 1.00
    n_ref = n_ref if isinstance(n_ref, (int, float)) and n_ref > 0 else 8.0
    w_max = w_max if isinstance(w_max, (int, float)) and w_max > 0 else 0.30
    rf = rf if isinstance(rf, (int, float)) else 0.045

    try:
        stage3_by_ticker = {r["Ticker"]: r for r in stage3_payload}
        daily_returns_df = pd.DataFrame(daily_returns_data).set_index('Date')

        common_tickers = [t for t in stage3_by_ticker if t in tailrisk["z"] and t in daily_returns_df.columns and t in crash_data["tickers"]]
        if len(common_tickers) < 2:
            return {"error": f"Za mało wspólnych tickerów między Stage 3 a danymi historycznymi ({len(common_tickers)}) — potrzeba min. 2."}

        daily_returns_df = daily_returns_df[common_tickers]

        # Estrada Downside Semi-Covariance (MAR=0, annualizowana, Tikhonov-regularyzowana) -- z tps_solver.py
        sigma_full = compute_estrada_matrix(daily_returns_df)

        # Discrete Crash-Overlap Matrix K -- już policzona w store-crash-matrices (Sekcja 3.3), tylko odtwarzamy DataFrame i reindeksujemy do common_tickers
        k_tickers = crash_data["tickers"]
        k_full_all = pd.DataFrame(crash_data["K"], index=k_tickers, columns=k_tickers)
        k_full = k_full_all.loc[common_tickers, common_tickers]

        mu_vec, p_vec = {}, {}
        for t in common_tickers:
            z = tailrisk["z"][t]
            p_vec[t] = compute_tail_penalty(z, lam)  # DIAGNOSTYKA ONLY -- solver tego nie widzi (decyzja d)
            upside = compute_composite_upside_row(stage3_by_ticker[t], gamma, kappa, n_ref, alpha=alpha)
            mu_vec[t] = upside["mu_i"]
        mu_vec = pd.Series(mu_vec)
        p_vec = pd.Series(p_vec)

        clusters_dict = {}
        for t in common_tickers:
            ck = stage3_by_ticker[t].get("Assigned Cluster", 1)
            clusters_dict.setdefault(ck, []).append(t)

        try:
            split_result = run_optimization_with_singleton_split(mu_vec, sigma_full, k_full, clusters_dict, lam, w_max=w_max, Rf=rf, nu=(nu if isinstance(nu, (int, float)) else 0.0))
        except (ValueError, RuntimeError) as e:
            return {"error": f"SLSQP (True Two-Stage + Singleton Split): {str(e)}"}

        final = split_result["final"]
        weights = final["weights"]

        eval_opt = evaluate_portfolio_performance(weights, daily_returns_df, is_log_returns=True, V0=100.0,
                                                      downside_cov_matrix=sigma_full, P_vec=p_vec, Rf=rf)
        delta_ann_map = semideviation_ann_from_matrix(sigma_full)

        # Standalone Asset TPS -- diagnostyka odłączona od solvera (decyzja e/6): "jak wyglądałaby
        # spółka w izolacji", NIE napędza już alokacji (to był Twój pierwotny "błąd metodologiczny").
        asset_tps = compute_asset_tps(mu_vec, p_vec, delta_ann_map, Rf=rf)

        cluster_of = {t: ck for ck, members in split_result["final_clusters_dict"].items() for t in members}

        # Diagnostyka Singleton Split dla UI (alert + multi-stage wykres) -- patrz render_singleton_split_panel.
        # Każdy wpis w history["weights"] może być None (przebieg był infeasible -- Stage 2 nigdy nie
        # wyprodukował wag) -- dla wizualizacji uzupełniamy TAKIE przebiegi wariantem "uncapped"
        # (w_max=1.0 na danym przebiegu g*/klastrach), żeby zawsze było co narysować, nie puste słupki.
        history_payload = []
        for h in split_result["history"]:
            weights_for_chart = h["weights"]
            if weights_for_chart is None:
                pass_clusters = {k: v for k, v in h["clusters_dict"].items()}
                g_series = pd.Series(h["g"])
                uncapped = run_stage2_inter_cluster_slsqp(mu_vec, sigma_full, k_full, pass_clusters, g_series, lam, w_max=1.0, Rf=rf)
                weights_for_chart = uncapped["weights"].to_dict()
            history_payload.append({
                "pass": h["pass"], "weights": weights_for_chart, "infeasible": h["infeasible"],
                "newly_promoted": h["newly_promoted"],
            })

        split_payload = {
            "split_triggered": split_result["split_triggered"],
            "promoted_tickers": split_result["promoted_tickers"],
            "n_passes": split_result["n_passes"],
            "history": history_payload,
        }

        return {
            "error": None,
            "tickers": common_tickers,
            "weights": weights.to_dict(),
            "g": final["g"].to_dict(),
            "asset_tps": asset_tps.to_dict(),
            "mu_i_map": mu_vec.to_dict(),
            "delta_ann_map": delta_ann_map.to_dict(),
            "cluster_of": cluster_of,
            "mu_p": final["mu_p"], "delta_p": final["delta_p"], "k_penalty_p": final["k_penalty_p"], "tps_p": final["tps_p"],
            "lam_used": lam, "nu_used": (nu if isinstance(nu, (int, float)) else 0.0), "rf_used": rf,
            "cdd_p": eval_opt["cdd_quantile"],
            "solver_success": final["success"], "solver_message": final["message"],
            "w_max_used": w_max,
            "split": split_payload,
        }
    except Exception as e:
        print("=" * 60)
        print("[STAGE 4B] CRITICAL ERROR - PEŁNY TRACEBACK:")
        traceback.print_exc()
        print("=" * 60)
        return {"error": f"Nieoczekiwany błąd solvera: {str(e)}"}


@app.callback(
    Output("stage4b-error-banner", "children"), Output("stage4b-kpi-row", "children"),
    Output("table-stage4b-weights", "data"), Output("graph-stage4b-donut", "figure"),
    Output("stage4b-formula-breakdown", "children"),
    Input("store-stage4b-results", "data"),
    prevent_initial_call=True
)
def render_stage4b_results(results):
    empty_fig = go.Figure()
    empty_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"])

    if not results or results.get("error"):
        msg = results.get("error") if results else "Brak danych."
        banner = html.Div(f"{msg}", style={"color": THEME["orange"], "fontWeight": "bold", "padding": "16px",
                                              "backgroundColor": THEME["bg_input"], "borderRadius": "12px", "marginBottom": "25px"})
        return banner, [], [], empty_fig, html.Div()

    tickers = results["tickers"]
    weights, g_map, tps_map, cluster_of = results["weights"], results["g"], results["asset_tps"], results["cluster_of"]
    promoted_tickers = set(results.get("split", {}).get("promoted_tickers", []))

    banner = html.Div()
    if not results.get("solver_success", True):
        banner = html.Div(f"Solver ostrzeżenie: {results.get('solver_message','')}", style={
            "color": THEME["orange"], "fontWeight": "bold", "padding": "14px", "backgroundColor": THEME["bg_input"],
            "borderRadius": "12px", "marginBottom": "25px", "fontSize": "12px"
        })

    # --- KPI strip (jedna zrosnieta belka, nie osobne plywajace karty) ---
    kpi_cards = build_kpi_strip([
        {"label": "OPTIMIZED EXPECTED RETURN (μ_P)", "value": f"{results['mu_p']*100:+.1f}%"},
        {"label": "ANNUALIZED DOWNSIDE RISK (δ_P)", "value": f"{results['delta_p']*100:.1f}%"},
        {"label": "10% TAIL RISK FLOOR (CDD 0.10,P)", "value": f"-{results['cdd_p']*100:.1f}%", "color": THEME["neg"]},
        {"label": "PORTFOLIO TPS SCORE (TPS_P)", "value": f"{results['tps_p']:.2f}", "color": THEME["accent"]},
    ])

    # --- Weights table ---
    # Promowane (auto-singleton) tickery dostają "★" w kolumnie Cluster + czerwono-pomarańczowy font
    # (style_data_conditional na filter_query 'contains "★"" w statycznej definicji DataTable) --
    # widać na pierwszy rzut oka, że dana spółka stanowi teraz autonomiczną klasę aktywów, bez
    # zgadywania po samej etykiecie klastra.
    table_rows = [{
        "Ticker": t, "Cluster": (f"★ {cluster_of.get(t, 1)}" if t in promoted_tickers else cluster_of.get(t, 1)),
        "g_i": g_map.get(t, 0.0), "w_i": weights.get(t, 0.0), "TPS_i": tps_map.get(t, 0.0)
    } for t in sorted(tickers, key=lambda x: weights.get(x, 0.0), reverse=True)]

    # --- Donut chart ---
    donut_tickers = [t for t in tickers if weights.get(t, 0.0) > 1e-6]
    donut_values = [weights[t] for t in donut_tickers]

    def _donut_color(t):
        ck = cluster_of.get(t, 1)
        try:
            return CLUSTER_PALETTE.get(((int(ck) - 1) % 6) + 1, "#888888")
        except (ValueError, TypeError):
            return THEME["orange"]  # non-numeric cluster key (e.g. "SINGLETON_MU") -- promoted singleton

    donut_colors = [_donut_color(t) for t in donut_tickers]
    donut_fig = go.Figure(data=[go.Pie(
        labels=donut_tickers, values=donut_values, hole=0.62, marker=dict(colors=donut_colors, line=dict(color=THEME["bg_card"], width=2)),
        textinfo='label+percent', textfont=dict(color=THEME["text_white"], size=11)
    )])
    donut_fig.update_layout(
        template="plotly_dark", height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, margin=dict(l=10, r=10, t=10, b=10), font_family=THEME["font"]
    )

    formula_breakdown = build_tps_formula_breakdown(
        mu_p=results.get("mu_p"), rf=results.get("rf_used", 0.045),
        sigma_p=results.get("delta_p"), k_p=results.get("k_penalty_p"),
        lam=results.get("lam_used", 0.0), nu=results.get("nu_used", 0.0),
        tps_p=results.get("tps_p"),
    )

    return banner, kpi_cards, table_rows, donut_fig, formula_breakdown


@app.callback(
    Output("panel-singleton-split", "children"), Output("panel-singleton-split", "style"),
    Input("store-stage4b-results", "data"), prevent_initial_call=True
)
def render_singleton_split_panel(results):
    """
    Sekcja "Dynamic Singleton Split" (Iterative Multi-Pass, patrz PROJECT_CONTEXT.md, rozszerzenie
    Sekcji 3.4): gdy w dowolnym wieloelementowym klastrze g_i przekracza w_max (lub finalna waga
    ląduje dokładnie na w_max, dusząc resztę klastra), dominująca spółka jest automatycznie
    wydzielana do własnego singletona i CAŁY pipeline (Stage1+Stage2) uruchamiany jest ponownie --
    powtarzane aż struktura klastrów się ustabilizuje (może to być więcej niż jedna runda, jeśli
    usunięcie jednego dominanta ujawni kolejnego -- "kaskadowa dominacja"). Ten panel pokazuje
    CAŁĄ sekwencję: alert z kolejnością promocji + wykres słupkowy ze wszystkimi przebiegami.
    """
    hidden = {"display": "none", "marginBottom": "30px"}
    if not results or results.get("error"):
        return [], hidden
    split = results.get("split", {})
    if not split.get("split_triggered"):
        return [], hidden

    history = split.get("history", [])
    w_max_used = results.get("w_max_used")
    w_max_str = f"{w_max_used*100:.0f}%" if isinstance(w_max_used, (int, float)) else "w_max"
    n_passes = split.get("n_passes", len(history))

    # Sekwencja promocji do alertu: "Pass 1: wydzielono MU -> Pass 2: wydzielono NVDA -> ..."
    promotion_steps = [f"Pass {h['pass']}: wydzielono {', '.join(h['newly_promoted'])}" for h in history if h.get("newly_promoted")]
    sequence_str = "  →  ".join(promotion_steps)

    alert = html.Div(style={
        "backgroundColor": "#2A1F0A", "border": f"1px solid {THEME['orange']}", "borderRadius": "14px",
        "padding": "18px 22px", "marginBottom": "20px"
    }, children=[
        html.Span(f"⚡ AUTO-PROMOCJA DO SINGLETONA ({n_passes} przebiegi)" if n_passes > 2 else "⚡ AUTO-PROMOCJA DO SINGLETONA",
                   style={"color": THEME["orange"], "fontWeight": "700", "fontSize": "12px", "letterSpacing": "0.5px"}),
        html.Div(
            f"Twardy limit w_max = {w_max_str} wymusił iteracyjne wydzielanie dominujących spółek do "
            f"osobnych mikro-klastrów, aż struktura się ustabilizowała: {sequence_str}",
            style={"color": THEME["text_white"], "fontSize": "12.5px", "marginTop": "8px", "lineHeight": "1.6"}
        ),
    ])

    # Wykres wieloetapowy: jeden słupek na ticker na KAŻDY przebieg -- pokazuje migrację kapitału
    # krok po kroku, nie tylko "przed/po". Ostatni (finalny, ustabilizowany) przebieg podświetlony
    # pomarańczowo, wcześniejsze -- gradient szarości.
    all_tickers = sorted(
        {t for h in history for t in h["weights"].keys()},
        key=lambda t: history[-1]["weights"].get(t, 0.0), reverse=True
    )
    n_hist = len(history)
    fig = go.Figure()
    for idx, h in enumerate(history):
        is_final = (idx == n_hist - 1)
        label = f"Pass {h['pass']} — Final" if is_final else f"Pass {h['pass']}" + (" (infeasible, uncapped)" if h["infeasible"] else "")
        # gradient: najstarsze przebiegi ciemnoszare, coraz jaśniejsze aż do finalnego pomarańczowego
        grey_level = 60 + int(90 * (idx / max(n_hist - 1, 1)))
        color = THEME["orange"] if is_final else f"rgb({grey_level},{grey_level},{grey_level+8})"
        fig.add_trace(go.Bar(name=label, x=all_tickers, y=[h["weights"].get(t, 0.0) for t in all_tickers], marker_color=color))

    fig.update_layout(
        barmode="group", template="plotly_dark", height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"],
        margin=dict(l=40, r=20, t=20, b=40), font_family=THEME["font"], yaxis=dict(tickformat=".0%", title="Waga"),
        legend=dict(orientation="h", y=1.15, font=dict(color=THEME["text_white"], size=10))
    )

    return [alert, dcc.Graph(figure=fig, config={"displayModeBar": False})], {"display": "block", "marginBottom": "30px"}


@app.callback(
    Output("snapshot-save-status", "children"), Output("snapshot-save-status-stage4", "children"),
    Output("store-snapshots-refresh", "data", allow_duplicate=True),
    Input("btn-save-snapshot", "n_clicks"), Input("btn-save-snapshot-stage4", "n_clicks"),
    State("input-snapshot-name", "value"), State("input-snapshot-name-stage4", "value"),
    State("store-stage4b-results", "data"),
    State("store-stage3-final-payload", "data"), State("store-stage4a-params", "data"),
    State("store-stage4a-tailrisk", "data"), State("store-snapshots-refresh", "data"),
    prevent_initial_call=True
)
def save_snapshot_callback(_n_sandbox, _n_stage4, name_sandbox, name_stage4, stage4b_results, stage3_payload, stage4a_params, tailrisk, counter):
    """
    Confirmed (2026-09-18): ten sam zapis wywoływalny z DWÓCH miejsc --
    przycisk w Sandbox (istniejący) i nowy przycisk bezpośrednio w Rebalance
    (Tab 4), zaraz po tym jak solver skończy liczyć, zamiast wymuszać
    przejście do innej zakładki tylko po to, żeby zapisać właśnie policzony
    portfel. Jedna funkcja, dwa Outputy statusu (po jednym na każdą
    zakładkę) -- ustalamy przez callback_context, który przycisk faktycznie
    kliknięto, i tej samej logiki/danych używamy niezależnie od źródła.
    """
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    from_stage4 = (trigger_id == "btn-save-snapshot-stage4")
    snapshot_name = name_stage4 if from_stage4 else name_sandbox

    def _both(msg):
        return (dash.no_update, msg) if from_stage4 else (msg, dash.no_update)

    if not stage4b_results or stage4b_results.get("error"):
        s1, s2 = _both(html.Span("Brak poprawnych wyników optymalizacji do zapisania.", style={"color": THEME["orange"]}))
        return s1, s2, dash.no_update
    if not stage3_payload:
        s1, s2 = _both(html.Span("Brak danych Stage 3 (fundamentals) do zapisania.", style={"color": THEME["orange"]}))
        return s1, s2, dash.no_update

    weights = stage4b_results["weights"]
    cluster_of = stage4b_results["cluster_of"]
    stage3_by_ticker = {r["Ticker"]: r for r in stage3_payload}

    # Entry price = ostatnia znana cena z Stage 3 (już policzona, bez dodatkowego live-fetchu) + SPY na potrzeby przyszłego benchmarku
    entry_prices = {t: stage3_by_ticker[t].get("Current Price (P0)", 0.0) for t in weights if t in stage3_by_ticker}
    spy_price = snap.fetch_current_prices(["SPY"])
    if "SPY" in spy_price:
        entry_prices["SPY"] = spy_price["SPY"]

    # Zamrażamy Z_TR,i z dnia zapisu -- to jedyny sposób, żeby suwak Lambda w sandboxie mógł
    # poprawnie różnicować karę P_i między spółkami bez ponownego liczenia CDD z 5 lat historii.
    z_scores = {t: tailrisk["z"][t] for t in weights if tailrisk and t in tailrisk.get("z", {})}

    try:
        record = snap.save_snapshot(
            snapshot_name=snapshot_name or "",
            parameters=stage4a_params or {},
            fundamental_inputs=stage3_payload,
            final_weights=weights,
            cluster_of=cluster_of,
            entry_prices=entry_prices,
            z_scores=z_scores
        )
    except Exception as e:
        s1, s2 = _both(html.Span(f"Błąd zapisu: {str(e)}", style={"color": THEME["orange"]}))
        return s1, s2, dash.no_update

    spy_note = "" if "SPY" in entry_prices else " (nie udało się pobrać ceny SPY na benchmark)"
    status = html.Span(f"Zapisano: \"{record['snapshot_name']}\" ({record['snapshot_id']}){spy_note}",
                        style={"color": THEME["accent"], "fontWeight": "bold"})
    s1, s2 = _both(status)
    return s1, s2, (counter or 0) + 1


@app.callback(
    Output("store-stage4a-params", "data"),
    [Input(f"input-{cfg['id']}", "value") for cfg in STAGE4A_PARAMS_CONFIG],
    prevent_initial_call=True
)
def bundle_stage4a_params(*values):
    return {cfg["id"]: values[i] for i, cfg in enumerate(STAGE4A_PARAMS_CONFIG)}

@app.callback(
    [Output(f"badge-{cfg['id']}", "children") for cfg in STAGE4A_PARAMS_CONFIG] +
    [Output(f"badge-{cfg['id']}", "style") for cfg in STAGE4A_PARAMS_CONFIG],
    [Input(f"input-{cfg['id']}", "value") for cfg in STAGE4A_PARAMS_CONFIG],
    prevent_initial_call=True
)
def validate_param_badges(*values):
    base_style = {"fontSize": "10px", "color": THEME["text_dim"], "marginTop": "8px", "padding": "3px 8px",
                  "border": f"1px solid {THEME['border']}", "borderRadius": "20px", "display": "inline-block"}
    children_out, style_out = [], []
    for i, cfg in enumerate(STAGE4A_PARAMS_CONFIG):
        val = values[i]
        label = f"[Recommended: {cfg['min']} – {cfg['max']}]"
        style = dict(base_style)
        if not isinstance(val, (int, float)) or val < cfg["min"] or val > cfg["max"]:
            label += "  — OUTSIDE RANGE"
            style["color"] = THEME["orange"]
            style["border"] = f"1px solid {THEME['orange']}"
        children_out.append(label)
        style_out.append(style)
    return children_out + style_out

# ---------------------------------------------------------------------------
# Relative Value -- nakladka po optymalizacji (2026-09-18, "Droga B",
# nastepnie 2026-09-18 naprawa powtarzalnosci + wykresy Z-score).
# Wylacznie diagnostyczna: NIE zmienia store-stage4b-results, NIE wplywa
# na to, co zapisze SAVE PORTFOLIO. Dziala WYLACZNIE na juz-policzonym,
# koncowym wektorze wag solvera.
#
# DWUETAPOWA architektura (naprawa zaobserwowanej niepowtarzalnosci wynikow
# miedzy klikanieciami -- kazde klikniecie robilo NOWE zapytanie sieciowe
# o 10 lat cen, wiec dwa "identyczne" uruchomienia moglo cicho dostac inne
# dane pod spodem, myloen z bledem w matematyce przechylenia, ktora
# faktycznie jest poprawna i deterministyczna -- zweryfikowane bezposrednio):
#   Etap A (drogi, btn-run-pair-overlay, background=True): pobiera 10Y RAZ,
#     uruchamia dwuetapowy test trwalosci + Maximum Weight Matching, cachuje
#     WYNIK (ktore pary, ich aktualny Z-score, historia Z-score do wykresu)
#     w store-pair-overlay-match-cache. Dobor par jest theta-niezmienniczy
#     (t-statystyka mechanizmu theta nie zalezy od thety), wiec ten etap
#     NIE musi sie powtarzac przy samej zmianie thety.
#   Etap B (tani, reaguje na zmiane theta LUB na nowy wynik Etapu A): czyta
#     cache, stosuje wzor przechylenia z aktualna theta, renderuje tabele +
#     wykresy Z-score. Zero pobierania danych, natychmiastowe.
# ---------------------------------------------------------------------------

def _build_pair_overlay_output(weights, match_cache, theta, w_max):
    return build_pair_overlay_output(weights, match_cache, theta, w_max, apply_tilts_to_matched_pairs)


@app.callback(
    Output("store-pair-overlay-match-cache", "data"), Output("pair-overlay-status", "children", allow_duplicate=True),
    Input("btn-run-pair-overlay", "n_clicks"),
    State("input-overlay-theta", "value"), State("store-stage4b-results", "data"),
    background=True, progress=[Output("pair-overlay-status", "children", allow_duplicate=True)],
    prevent_initial_call=True,
)
def find_pair_overlay_matches(set_progress, _n_clicks, theta, stage4b_results):
    """Etap A (drogi) -- patrz komentarz modulu powyzej. Wykonuje sie TYLKO
    na klikniecie przycisku, NIGDY na sama zmiane thety."""
    theta = theta or 0.15

    if not stage4b_results or stage4b_results.get("error"):
        return dash.no_update, html.Div("Brak poprawnych wyników solvera -- uruchom optymalizację powyżej.", style={"color": THEME["orange"]})

    weights = stage4b_results["weights"]
    selected_tickers = [t for t, w in weights.items() if w and w > 1e-9]
    if len(selected_tickers) < 2:
        return dash.no_update, html.Div("Za mało spółek z dodatnią wagą do sprawdzenia par.", style={"color": THEME["orange"]})

    set_progress([f"Pobieram niezależnie 10 lat historii cen dla {len(selected_tickers)} wybranych spółek..."])
    prices_df, valid_tickers = fetch_universe_prices(selected_tickers, period="10y")
    if prices_df.empty or len(valid_tickers) < 2:
        return dash.no_update, html.Div("Nie udało się pobrać wystarczająco długiej (10-letniej) historii cenowej.", style={"color": THEME["orange"]})

    def _report_progress(label, i, total):
        if label == "Etap 1: tanie sito":
            set_progress([f"Tanie sito theta (kanonizacja): kombinacja {i}/{total}..."])
        else:
            set_progress([f"Pełny test trwałości ({label}): para {i}/{total}..."])

    set_progress([f"Sprawdzam pary wśród {len(valid_tickers)} spółek z dodatnią wagą..."])
    match_data = find_matched_pairs_for_overlay(weights, prices_df, theta=theta, on_progress=_report_progress)
    status = html.Div(
        f"Dobór par zakończony -- {match_data['n_selected']} spółek, {match_data['n_qualifying']} par kwalifikujących się, "
        f"{match_data['n_eligible']} dopuszczalnych, {len(match_data['matched_pairs_info'])} dopasowanych. "
        f"Teraz możesz dowolnie zmieniać THETA bez ponownego pobierania danych.",
        style={"color": THEME["accent"]}
    )
    return match_data, status


@app.callback(
    Output("pair-overlay-results", "children"), Output("pair-overlay-status", "children", allow_duplicate=True),
    Input("store-pair-overlay-match-cache", "data"), Input("input-overlay-theta", "value"),
    State("store-stage4b-results", "data"), State("input-wmax", "value"),
    prevent_initial_call=True,
)
def render_pair_overlay_tilts(match_cache, theta, stage4b_results, w_max):
    """Etap B (tani) -- reaguje na kazda zmiane thety LUB na swiezy wynik
    Etapu A, zero pobierania danych. Patrz komentarz modulu powyzej."""
    theta = theta or 0.15
    w_max = w_max if isinstance(w_max, (int, float)) and w_max > 0 else 0.30
    if not stage4b_results or stage4b_results.get("error"):
        return dash.no_update, dash.no_update
    weights = stage4b_results["weights"]
    return _build_pair_overlay_output(weights, match_cache, theta, w_max)