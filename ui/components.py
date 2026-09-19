"""
ui/components.py
==================
Reusable, generic UI building blocks shared across multiple tabs: color-blend
DataTable cell styling, KPI cards, and the Stage 4 parameter input card
(driven by STAGE4A_PARAMS_CONFIG).

Moved out of quant_terminal.py (Etap 0 architecture split, PROJECT_CONTEXT.md)
with NO behavior change.
"""
import re

import numpy as np
import pandas as pd
from dash import dcc, html, dash_table
import plotly.graph_objects as go

from ui.theme import THEME

TICKER_PATTERN = re.compile(r'^[A-Z0-9]{1,10}(\.[A-Z]{1,4})?$')


def parse_single_ticker_input(raw_text):
    """
    Validates a raw "add new ticker" text-input value is exactly ONE
    plausible ticker -- shared by ui/tab1_market_data.py's "+ DODAJ DO
    UNIWERSUM" and ui/module_research.py's "+ ŚLEDŹ", both of which
    previously accepted whatever was typed as a single literal ticker
    string with no validation.

    Real bug found in testing (2026-09-08): typing "AVGO, CRDO" (meaning to
    add Broadcom and Credo, or simply a typo) was silently accepted as ONE
    ticker string "AVGO, CRDO" -- yfinance's lenient `.info` search loosely
    matched it to Broadcom's profile, so the Name/Sector auto-fill looked
    plausible, while the stored `Ticker` field kept the literal
    "AVGO, CRDO" including the comma and second symbol, as if ", CRDO" were
    part of the ticker itself.

    Splits on common separators (comma, semicolon, whitespace) and REJECTS
    (rather than silently taking the first token, or concatenating them)
    if more than one token results -- this project's "+" buttons only ever
    add one company at a time by design, so multiple tokens signal a
    mistake or a misunderstanding of the field, not a batch-add request.

    Also rejects a single token that doesn't match a plausible ticker shape
    (`TICKER_PATTERN`: letters/digits, with an optional dot-suffix for
    non-US exchanges already present in this project's universe, e.g.
    "005930.KS", "000660.KS", "LYC.AX").

    Returns (ticker: str, None) on success, or (None, error_message: str)
    otherwise -- never raises.
    """
    if not raw_text or not raw_text.strip():
        return None, "Wpisz ticker."

    tokens = [t for t in re.split(r'[,;\s]+', raw_text.strip()) if t]
    if len(tokens) > 1:
        return None, f"Wpisz tylko jeden ticker na raz (wykryto {len(tokens)}: {', '.join(tokens)})."

    ticker = tokens[0].upper()
    if not TICKER_PATTERN.match(ticker):
        return None, f"'{ticker}' nie wygląda na poprawny ticker."

    return ticker, None

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return np.array([int(hex_str[i:i+2], 16) for i in (0, 2, 4)])

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

# ---------------------------------------------------------------------------
# Współdzielone style dash_table.DataTable -- pełna siatka linii (border na
# każdej krawędzi komórki, jak w arkuszu kalkulacyjnym / IBKR / Bloomberg),
# naprzemienne cieniowanie wierszy, tabular-nums dla wyrównania cyfr w kolumnie.
# Każde wywołanie DataTable powinno korzystać z tych trzech funkcji zamiast
# powtarzać hardkodowane hex-y (`#0B0B0E`, `#222230` itp.) -- to była jedyna
# rzecz, przez którą design "Institutional, softened" był rozjechany między
# tabelami przed tą zmianą.
# ---------------------------------------------------------------------------

def datatable_style_header():
    return {
        "backgroundColor": THEME["bg_head"], "color": THEME["text_label"], "fontWeight": "600",
        "border": f"1px solid {THEME['border_strong']}", "padding": "9px 12px", "fontSize": "11px",
    }

def datatable_style_cell():
    return {
        "padding": "8px 12px", "textAlign": "center", "fontSize": "12.5px",
        "border": f"1px solid {THEME['border']}", "fontVariantNumeric": "tabular-nums",
    }

def datatable_style_data():
    return {"backgroundColor": THEME["bg_card"], "color": THEME["text_white"], "border": f"1px solid {THEME['border']}"}

def datatable_row_alt_rule():
    """Prepend this to a table's `style_data_conditional` list for alternating-row
    shading -- put it FIRST so column-specific color rules (e.g. Delta > 0 -> green)
    still apply on top, since this rule only ever touches backgroundColor."""
    return {"if": {"row_index": "odd"}, "backgroundColor": THEME["bg_row_alt"]}

def generate_tws_matrix_styles(df, tickers):
    """
    Colors each cell of the correlation DataTable per the standard
    -1=red / 0=neutral(background) / +1=blue convention, using the SAME
    gentle, capped-saturation blend as the Plotly matrix heatmaps
    (ui/theme.py's MATRIX_COLORSCALE) -- correction (2026-09-07, second
    follow-up): this previously blended negative values toward
    THEME["orange"] (amber), not red, which contradicted the intended
    convention and this table's own color story diverging from the Plotly
    heatmaps using the correct red/blue scale.

    MAX_BLEND caps how far toward full red/blue a cell can go even at
    |val|=1 -- kept well short of 1.0 so cells never hit raw THEME["neg"]/
    THEME["accent"] saturation, matching the "gentler colors, numbers must
    stay readable" request.
    """
    MAX_BLEND = 0.55
    styles = []
    c_bg, c_pos, c_neg = hex_to_rgb(THEME["bg_card"]), hex_to_rgb(THEME["accent"]), hex_to_rgb(THEME["neg"])
    for t in tickers:
        for _, row in df.iterrows():
            val = row[t]
            if pd.isna(val): continue
            blend = min(abs(val), 1.0) * MAX_BLEND
            rgb_mixed = c_bg + (c_pos - c_bg) * blend if val >= 0 else c_bg + (c_neg - c_bg) * blend
            styles.append({'if': {'filter_query': f'{{Ticker}} eq "{row["Ticker"]}"', 'column_id': t}, 'backgroundColor': rgb_to_hex(np.clip(rgb_mixed, 0, 255)), 'color': '#FFFFFF'})
    return styles

STAGE4A_PARAMS_CONFIG = [
    {"id": "alpha", "name": "Alpha Blend", "symbol": "α", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
     "comment": "Growth/Upside Blend (Section 4.2, Etap 2). α=0.0 → pure analyst target-price consensus. α=0.5 → equal blend. α=1.0 → pure fundamental EPS growth (the analyst dispersion penalty γ drops out entirely at α=1.0, since it only ever discounts the target-price branch). Analyst coverage confidence A(N_i) still discounts the result at any α — coverage depth is treated as a general forecast-quality signal, not specific to target prices."},
    {"id": "lambda", "name": "Lambda", "symbol": "λ", "default": 3.0, "min": 0.1, "max": 25.0, "step": 0.1,
     "comment": "Crash-overlap penalty sensitivity: exp(λ·√(wᵀKw)). Rescaled for the K-based quadratic penalty (Section 3.3/3.4) — √(wᵀKw) typically runs ≈0.05–0.20, so λ needs a much wider range than the old per-asset Z-score exponent did to have a comparable effect."},
    {"id": "nu", "name": "Nu (wrażliwość zmienności)", "symbol": "ν", "default": 0.0, "min": -5.0, "max": 15.0, "step": 0.5,
     "comment": "Dodatkowy wykładniczy mnożnik na PORTFELOWEJ zmienności Estrady: exp(ν·√(wᵀΣ_εw)), obok istniejącego λ na K. Przy ν=0 wzór wraca dokładnie do poprzedniej wersji. ν>0 dokłada karę za zmienność silniejszą niż sam pierwiastek Sortino już daje; ν<0 PREMIUJE zmienność zamiast ją karać (teza Part I: w hiper-growth zmienność bywa górną, korzystną skośnością, nie czystym ryzykiem)."},
    {"id": "gamma", "name": "Gamma", "symbol": "γ", "default": 1.50, "min": 0.10, "max": 5.00, "step": 0.10,
     "comment": "Analyst range spread penalty. Disincentivizes stocks with wide target price disagreements."},
    {"id": "kappa", "name": "Kappa", "symbol": "κ", "default": 1.00, "min": 0.00, "max": 5.00, "step": 0.10,
     "comment": "Sensitivity to 90-day EPS consensus revisions (short-term earnings momentum)."},
    {"id": "nref", "name": "N_ref", "symbol": "N_ref", "default": 8.0, "min": 3, "max": 30,
     "comment": "Reference analyst coverage threshold for maximum confidence factor A_i."},
    {"id": "wmax", "name": "Max Asset Weight", "symbol": "w_max", "default": 0.30, "min": 0.05, "max": 1.00, "step": 0.05,
     "comment": "Hard single-stock concentration cap (e.g., 0.30 = max 30% weight per stock). Enforced as a bound directly in the Stage 2 SLSQP; if Stage 1's intra-cluster concentration makes this infeasible, the Singleton Split procedure automatically carves the dominant asset(s) into their own capped micro-cluster — see the alert panel below the results."},
    {"id": "rf", "name": "Risk-Free Rate", "symbol": "R_f", "default": 0.045, "min": 0.0, "max": 0.25, "step": 0.005,
     "comment": "Annualized risk-free / qualitative hurdle rate. Ręczne wejście, brak automatycznego pobierania ^TNX (zgodnie z zasadą 'dane manualne')."},
]

def build_param_card(cfg):
    step = cfg.get("step", 0.01)
    return html.Div(style={
        "backgroundColor": THEME["bg_input"], "border": f"1px solid {THEME['border']}", "borderRadius": "6px",
        "padding": "14px 16px", "flex": "1 1 220px", "minWidth": "220px"
    }, children=[
        html.Div([
            html.Span(cfg["name"], style={"fontSize": "12px", "fontWeight": "600", "color": THEME["text_white"]}),
            html.Span(f"  {cfg['symbol']}", style={"fontSize": "12px", "color": THEME["accent"], "fontWeight": "600", "marginLeft": "4px"})
        ], style={"marginBottom": "8px"}),
        dcc.Input(id=f"input-{cfg['id']}", type="number", value=cfg["default"], step=step,
                   style={"width": "100%", "padding": "8px 10px", "backgroundColor": THEME["bg_base"], "border": f"1px solid {THEME['border']}",
                          "borderRadius": "4px", "color": THEME["text_white"], "fontSize": "14px", "fontWeight": "600",
                          "fontVariantNumeric": "tabular-nums", "boxSizing": "border-box"}),
        html.Div(id=f"badge-{cfg['id']}", children=f"[Recommended: {cfg['min']} – {cfg['max']}]", style={
            "fontSize": "10px", "color": THEME["text_label"], "marginTop": "8px", "padding": "2px 7px",
            "border": f"1px solid {THEME['border']}", "borderRadius": "3px", "display": "inline-block"
        }),
        html.Div(cfg["comment"], style={"fontSize": "11px", "color": THEME["text_dim"], "marginTop": "9px", "lineHeight": "1.5"})
    ])

def build_kpi_card(label, value_str, sub_str="", color=None):
    color = color or THEME["text_white"]
    return html.Div(style={
        "backgroundColor": THEME["bg_input"], "border": f"1px solid {THEME['border']}", "borderRadius": "6px", "padding": "16px 18px"
    }, children=[
        html.Div(label, style={"fontSize": "11px", "color": THEME["text_label"], "fontWeight": "600", "marginBottom": "8px", "letterSpacing": "0.02em"}),
        html.Div(value_str, style={"fontSize": "22px", "fontWeight": "600", "color": color, "fontVariantNumeric": "tabular-nums", "letterSpacing": "-0.01em"}),
        html.Div(sub_str, style={"fontSize": "11px", "color": THEME["text_dim"], "marginTop": "5px"})
    ])

def build_kpi_strip(items):
    """
    Renders a list of KPIs as ONE connected instrument strip (single outer border,
    thin vertical dividers between items) instead of separately-boxed cards in a
    gapped grid -- matches the approved "Institutional, softened" mockup
    (2026-09-07 design conversation), where the KPI row reads as one cohesive
    panel rather than floating pills.

    `items`: list of dicts, each with keys:
        "label" (str), "value" (str), "sub" (str, optional), "color" (str hex, optional)

    The wrapping Output div in ui/layout.py must NOT impose its own grid/gap
    styling on top of this -- build_kpi_strip returns one fully self-contained
    flex container, not a list of children expecting a grid wrapper.
    """
    n = len(items)
    children = []
    for i, it in enumerate(items):
        is_last = (i == n - 1)
        children.append(html.Div(style={
            "padding": "14px 20px", "borderRight": "none" if is_last else f"1px solid {THEME['border']}",
            "flex": "1", "minWidth": "0",
        }, children=[
            html.Div(it["label"], style={"fontSize": "11px", "color": THEME["text_label"], "fontWeight": "600", "marginBottom": "7px", "letterSpacing": "0.01em"}),
            html.Div(it["value"], style={"fontSize": "20px", "fontWeight": "600", "color": it.get("color") or THEME["text_white"], "fontVariantNumeric": "tabular-nums", "letterSpacing": "-0.01em"}),
            html.Div(it.get("sub", ""), style={"fontSize": "10.5px", "color": THEME["text_dim"], "marginTop": "4px"}),
        ]))
    return html.Div(style={
        "display": "flex", "backgroundColor": THEME["bg_card"], "border": f"1px solid {THEME['border_strong']}", "borderRadius": "4px",
    }, children=children)


# ============================================================
# STAGE 3 — shared table-shape constants (used by both ui/layout.py, which
# defines the DataTable, and ui/tab2_fundamentals.py, which populates it)
# ============================================================

STAGE3_BASELINE_MODELS = {
    "semicov_ward": ("semicov", "ward"), "semicov_single": ("semicov", "single"), "semicov_complete": ("semicov", "complete"),
    "std_ward": ("standard", "ward"), "std_single": ("standard", "single"), "std_complete": ("standard", "complete"),
}

STAGE3_COLUMNS = [
    {"name": "Ticker", "id": "Ticker", "editable": False},
    {"name": "Assigned Cluster", "id": "Assigned Cluster", "type": "numeric", "format": {"specifier": "d"}, "editable": True},
    {"name": "Current Price (P₀)", "id": "Current Price (P0)", "type": "numeric", "format": {"specifier": ".2f"}, "editable": True},
    {"name": "Target Consensus (Ti)", "id": "Target Consensus (Ti)", "type": "numeric", "format": {"specifier": ".2f"}, "editable": True},
    {"name": "Target High (T_high)", "id": "Target High (T_high)", "type": "numeric", "format": {"specifier": ".2f"}, "editable": True},
    {"name": "Target Low (T_low)", "id": "Target Low (T_low)", "type": "numeric", "format": {"specifier": ".2f"}, "editable": True},
    {"name": "Analyst Coverage (Ni)", "id": "Analyst Coverage (Ni)", "type": "numeric", "format": {"specifier": "d"}, "editable": True},
    {"name": "EPS 2Y CAGR (Gi,2Y) [%]", "id": "EPS 2Y CAGR (Gi,2Y)", "type": "numeric", "format": {"specifier": ".2f"}, "editable": True},
    {"name": "90d EPS Revision (ΔEPS90d) [%]", "id": "90d EPS Revision (ΔEPS90d)", "type": "numeric", "format": {"specifier": ".2f"}, "editable": True},
]
STAGE3_FUNDAMENTAL_COLS = ["Current Price (P0)", "Target Consensus (Ti)", "Target High (T_high)", "Target Low (T_low)",
                           "Analyst Coverage (Ni)", "EPS 2Y CAGR (Gi,2Y)", "90d EPS Revision (ΔEPS90d)"]
# UWAGA: EPS 2Y CAGR i 90d EPS Revision są WPISYWANE i PRZECHOWYWANE jako liczby całe/procenty
# (np. 55 = 55%, -3 = -3%). Podział przez 100.0 następuje WYŁĄCZNIE wewnątrz compute_composite_upside_row,
# nigdy w samej tabeli — dzięki temu nie trzeba wpisywać 0.55 zamiast 55.

# ---------------------------------------------------------------------------
# Relative Value -- nakladka po optymalizacji: wspolne budowniczy UI dla
# Rebalance (Tab 4) i Sandbox (Tab 5), zeby oba wygladaly i dzialaly
# identycznie (2026-09-18).
# ---------------------------------------------------------------------------

def build_pair_zscore_figure(pair_info, today_label="dziś"):
    """Wykres Z-score dla jednej dopasowanej pary -- linia historyczna,
    linia zerowa, i przerywana linia + adnotacja na aktualnym Z-score.
    `today_label` pozwala Sandboxowi podpisac to jako "dzien zapisu"
    zamiast "dzis", skoro w Sandboxie caly ten punkt jest z definicji
    zamrozony na dacie utworzenia snapshotu, nie na biezacej dacie
    (2026-09-19, naprawa look-ahead bias)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pair_info["zscore_dates"], y=pair_info["zscore_values"], mode="lines",
        line=dict(color=THEME["accent"], width=1.5), name="Z-score"
    ))
    fig.add_hline(y=0, line=dict(color=THEME["text_dim"], width=1, dash="dot"))
    fig.add_hline(y=pair_info["current_z"], line=dict(color=THEME["orange"], width=1.5, dash="dash"))
    fig.add_annotation(
        x=pair_info["zscore_dates"][-1], y=pair_info["current_z"],
        text=f"{today_label}: Z={pair_info['current_z']:.2f}", showarrow=True, arrowhead=2,
        font=dict(color=THEME["orange"], size=10), ax=-60, ay=-25
    )
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"],
        height=220, margin=dict(l=40, r=20, t=30, b=30),
        title=dict(text=f"{pair_info['ticker_a']} / {pair_info['ticker_b']} -- Z-score", font=dict(size=12, color=THEME["text_white"])),
        xaxis=dict(showgrid=False, tickfont=dict(size=9, color=THEME["text_dim"])),
        yaxis=dict(title="Z-score", showgrid=True, gridcolor="#1E1E28", tickfont=dict(size=9, color=THEME["text_dim"])),
        showlegend=False,
    )
    return fig


def build_pair_weight_history_figure(pair_info, theta, today_label="dziś"):
    """
    Udzial wagi tickera A w parze W CZASIE, WYLICZONY z juz-zbuforowanej
    historii Z-score (zero dodatkowego liczenia/pobierania -- ten sam wzor
    0.5+0.5*theta*tanh(-Z), zastosowany do calej historii Z, nie tylko do
    ostatniego punktu). Odpowiada wprost na pytanie "jak w czasie zmienialby
    sie podzial tej pary" -- reaguje na biezaca theta natychmiast, bo caly
    wklad to Z-score, ktore juz mamy (2026-09-19)."""
    z_arr = np.array(pair_info["zscore_values"])
    weight_a_history = 0.5 + 0.5 * theta * np.tanh(-z_arr)
    current_weight_a = float(weight_a_history[-1]) if len(weight_a_history) else 0.5

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pair_info["zscore_dates"], y=weight_a_history * 100, mode="lines",
        line=dict(color=THEME["pos"] if "pos" in THEME else "#00C853", width=1.5), name=f"Udział {pair_info['ticker_a']}"
    ))
    fig.add_hline(y=50, line=dict(color=THEME["text_dim"], width=1, dash="dot"))
    fig.add_annotation(
        x=pair_info["zscore_dates"][-1], y=current_weight_a * 100,
        text=f"{today_label}: {pair_info['ticker_a']}={current_weight_a*100:.1f}%", showarrow=True, arrowhead=2,
        font=dict(color=THEME["orange"], size=10), ax=-60, ay=-25
    )
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"],
        height=220, margin=dict(l=40, r=20, t=30, b=30),
        title=dict(text=f"{pair_info['ticker_a']} / {pair_info['ticker_b']} -- udział wagi w parze (θ={theta})", font=dict(size=12, color=THEME["text_white"])),
        xaxis=dict(showgrid=False, tickfont=dict(size=9, color=THEME["text_dim"])),
        yaxis=dict(title=f"Udział {pair_info['ticker_a']} (%)", range=[0, 100], showgrid=True, gridcolor="#1E1E28", tickfont=dict(size=9, color=THEME["text_dim"])),
        showlegend=False,
    )
    return fig


def build_pair_overlay_output(weights, match_cache, theta, w_max, apply_tilts_fn, today_label="dziś"):
    """
    Tabela przed/po + wykresy (Z-score ORAZ udzial wagi w czasie) +
    diagnostyka utraconej alternatywy -- wspolne dla Rebalance i Sandbox.
    `apply_tilts_fn` to engine.pairs.apply_tilts_to_matched_pairs
    (wstrzykniete jako argument, zeby components.py nie musial importowac
    engine/ -- utrzymuje UI/engine jako oddzielne warstwy). `today_label`:
    "dziś" w Rebalansie (analiza faktycznie jest na dzis), "dzień zapisu"
    w Sandboxie (analiza jest zamrozona na dacie utworzenia snapshotu,
    2026-09-19 naprawa look-ahead bias -- patrz ui/tab5_sandbox.py).
    """
    if not match_cache or match_cache.get("n_selected", 0) < 2:
        return html.Div(), html.Div("Brak wyniku -- najpierw uruchom \"ZNAJDŹ PARY\".", style={"color": THEME["orange"]})

    tilt_result = apply_tilts_fn(weights, match_cache["matched_pairs_info"], theta=theta, wmax=w_max)
    selected_tickers = [t for t, w in weights.items() if w and w > 1e-9]

    rows = []
    for t in selected_tickers:
        before = weights.get(t, 0.0)
        after = tilt_result["weights"].get(t, before)
        rows.append({"Ticker": t, "Waga przed": round(before, 4), "Waga po": round(after, 4), "Zmiana": round(after - before, 4)})

    table = dash_table.DataTable(
        columns=[
            {"name": "Ticker", "id": "Ticker"},
            {"name": "Waga przed", "id": "Waga przed", "type": "numeric", "format": {"specifier": ".4f"}},
            {"name": "Waga po", "id": "Waga po", "type": "numeric", "format": {"specifier": ".4f"}},
            {"name": "Zmiana", "id": "Zmiana", "type": "numeric", "format": {"specifier": "+.4f"}},
        ],
        data=rows, sort_action="native", page_size=25,
        style_header=datatable_style_header(), style_data=datatable_style_data(), style_cell=datatable_style_cell(),
        style_data_conditional=[
            datatable_row_alt_rule(),
            {"if": {"filter_query": "{Zmiana} > 0", "column_id": "Zmiana"}, "color": THEME["accent"], "fontWeight": "bold"},
            {"if": {"filter_query": "{Zmiana} < 0", "column_id": "Zmiana"}, "color": THEME["orange"], "fontWeight": "bold"},
        ],
    )

    wmax_note = html.Div(
        "Nakładka IGNORUJE w_max całkowicie -- ma własną mechanikę (mnożnikowo-przeskalowaną, nie ograniczoną limitem koncentracji "
        "solvera). Przy wystarczająco silnym sygnale para może wylądować ze znacznie wyższą koncentracją niż standardowy limit portfela.",
        style={"fontSize": "10px", "color": THEME["text_dim"], "backgroundColor": "rgba(255,255,255,0.03)",
               "padding": "8px 10px", "borderRadius": "4px", "marginTop": "10px", "lineHeight": "1.5"}
    ) if match_cache["matched_pairs_info"] else html.Div()

    network_section = html.Div()
    if match_cache.get("qualifying_pairs"):
        net_fig, matrix_fig = build_pair_network_and_matrix(
            match_cache.get("all_selected_tickers", selected_tickers),
            match_cache["qualifying_pairs"], match_cache.get("eligible_pairs", []), match_cache["matched_pairs_info"]
        )
        network_section = html.Div([
            html.Div("SIEĆ I MACIERZ T-STATYSTYK (zielone/pogrubione = wybrana para, przerywane szare = dopuszczalna ale przegrana, cienkie blade = tylko kwalifikuje się):",
                     style={"fontSize": "10.5px", "fontWeight": "bold", "color": THEME["text_dim"], "marginTop": "18px", "marginBottom": "8px"}),
            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"}, children=[
                dcc.Graph(figure=net_fig, config={"displayModeBar": False}, style={"height": "360px"}),
                dcc.Graph(figure=matrix_fig, config={"displayModeBar": False}, style={"height": "360px"}),
            ])
        ])

    charts = html.Div()
    if match_cache["matched_pairs_info"]:
        chart_divs = []
        for p in match_cache["matched_pairs_info"]:
            chart_divs.append(dcc.Graph(figure=build_pair_zscore_figure(p, today_label=today_label), config={"displayModeBar": False}))
            chart_divs.append(dcc.Graph(figure=build_pair_weight_history_figure(p, theta, today_label=today_label), config={"displayModeBar": False}))
        t_stats_line = ", ".join(f"{p['ticker_a']}/{p['ticker_b']}: t={p.get('t_statistic', '?')}" for p in match_cache["matched_pairs_info"])
        charts = html.Div([
            html.Div(f"KAŻDA DOPASOWANA PARA: Z-score i wynikający z niego udział wagi w czasie (kontekst ostatnich ~2 lat, linia przerywana = {today_label}). "
                     f"t-statystyka: {t_stats_line}",
                     style={"fontSize": "10.5px", "fontWeight": "bold", "color": THEME["text_dim"], "marginTop": "18px", "marginBottom": "8px"}),
            html.Div(chart_divs, style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))", "gap": "12px"})
        ])

    blocking_info = html.Div()
    if match_cache["unmatched_with_alternative"]:
        items = [html.Li(f"{u['ticker']}: nieużyta relacja z {u['alternative_ticker']} (najlepsza t-statystyka={u['best_t_statystyka']})")
                 for u in match_cache["unmatched_with_alternative"]]
        blocking_info = html.Div([
            html.Div("SPÓŁKI BEZ PARY, ALE Z NIEUŻYTĄ, KWALIFIKUJĄCĄ SIĘ ALTERNATYWĄ (informacyjnie, Droga B -- bez automatycznego działania):",
                     style={"fontSize": "10.5px", "fontWeight": "bold", "color": THEME["warn"], "marginTop": "18px", "marginBottom": "8px"}),
            html.Ul(items, style={"fontSize": "11px", "color": THEME["text_dim"]}),
        ])

    as_of_note = f" (dane wyłącznie do {match_cache['as_of_date']})" if match_cache.get("as_of_date") else ""
    status = html.Div(
        f"{match_cache['n_selected']} spółek z dodatnią wagą{as_of_note} -- {match_cache['n_qualifying']} par kwalifikujących się, "
        f"{match_cache['n_eligible']} dopuszczalnych do skojarzenia, {len(tilt_result['applied_pairs'])} faktycznie zastosowanych "
        f"(theta={theta}).",
        style={"color": THEME["accent"]}
    )
    return html.Div([table, wmax_note, network_section, charts, blocking_info]), status


# ---------------------------------------------------------------------------
# Siec grafow + macierz t-statystyk -- wspolny budowniczy (2026-09-19),
# wydzielony ze wzorca juz uzywanego w ui/tab1_market_data.py::run_pair_matching_analysis,
# zeby nakladka Relative Value (Rebalance/Sandbox) mogla pokazac dokladnie
# taka sama, trojpoziomowa wizualizacje (wybrana/dopuszczalna-przegrana/
# tylko-kwalifikujaca-sie), zamiast tylko tabeli i osobnych wykresow Z-score.
# ---------------------------------------------------------------------------

def _circular_layout_generic(nodes):
    """Deterministyczny uklad kolowy -- identyczny wzorzec co
    ui/tab1_market_data.py::_circular_layout, zdublowany tutaj celowo (mala,
    samodzielna funkcja), zeby components.py nie musial importowac z ui/tab1_market_data.py
    (odwrocilo by to kierunek zaleznosci warstwy UI)."""
    n = len(nodes)
    positions = {}
    for i, node in enumerate(nodes):
        angle = 2 * np.pi * i / n - np.pi / 2
        positions[node] = (np.cos(angle), np.sin(angle))
    return positions


def build_pair_network_and_matrix(all_selected_tickers, qualifying_pairs, eligible_pairs, matched_pairs_info):
    """
    Buduje (network_fig, matrix_fig) w dokladnie tym samym stylu co panel
    doboru par w Rebalansie/Stage 1 (3 poziomy: wybrana para / dopuszczalna
    ale przegrana w skojarzeniu / tylko kwalifikujaca sie). Parametry to
    juz-gotowe listy dictow z engine.pairs.find_matched_pairs_for_overlay
    (qualifying_pairs, eligible_pairs, matched_pairs_info), nie surowe
    DataFrame'y -- components.py nie zalezy od engine/.
    """
    empty_fig = go.Figure()
    empty_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"])
    if not qualifying_pairs:
        return empty_fig, empty_fig

    matched_edge_set = {(p["ticker_a"], p["ticker_b"]) for p in matched_pairs_info}
    matched_tickers = {t for pair in matched_edge_set for t in pair}
    eligible_edge_set = {(p["ticker_a"], p["ticker_b"]) for p in eligible_pairs}
    all_qualifying_edges = [(p["ticker_a"], p["ticker_b"], p["t_statistic"]) for p in qualifying_pairs]
    graph_nodes = sorted({t for a, b, _w in all_qualifying_edges for t in (a, b)})

    def _edge_tier(a, b):
        if (a, b) in matched_edge_set or (b, a) in matched_edge_set:
            return "selected"
        if (a, b) in eligible_edge_set or (b, a) in eligible_edge_set:
            return "eligible"
        return "qualifies_only"

    TIER_STYLE = {
        "selected": dict(color=THEME["pos"] if "pos" in THEME else "#00C853", width=4, dash="solid"),
        "eligible": dict(color=THEME["border_strong"], width=1.5, dash="dot"),
        "qualifies_only": dict(color=THEME["border"], width=0.75, dash="dot"),
    }

    positions = _circular_layout_generic(graph_nodes)
    net_fig = go.Figure()
    for a, b, w in all_qualifying_edges:
        tier = _edge_tier(a, b)
        style = TIER_STYLE[tier]
        x0, y0 = positions[a]; x1, y1 = positions[b]
        net_fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(color=style["color"], width=style["width"], dash=style["dash"]),
            hoverinfo="text", text=f"{a}/{b}: t={w:.3f} ({tier})", showlegend=False,
        ))
    node_x = [positions[n][0] for n in graph_nodes]
    node_y = [positions[n][1] for n in graph_nodes]
    node_colors = [(THEME["pos"] if "pos" in THEME else "#00C853") if n in matched_tickers else THEME["text_dim"] for n in graph_nodes]
    net_fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=graph_nodes, textposition="middle center",
        marker=dict(size=34, color=node_colors, line=dict(width=1, color=THEME["bg_base"])),
        textfont=dict(size=9, color=THEME["bg_base"]), showlegend=False, hoverinfo="text",
    ))
    net_fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"],
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(visible=False, range=[-1.3, 1.3]), yaxis=dict(visible=False, range=[-1.3, 1.3]),
    )

    matrix_tickers = graph_nodes
    z = np.full((len(matrix_tickers), len(matrix_tickers)), np.nan)
    idx = {t: i for i, t in enumerate(matrix_tickers)}
    for a, b, w in all_qualifying_edges:
        z[idx[a]][idx[b]] = w
        z[idx[b]][idx[a]] = w
    matrix_fig = go.Figure(data=go.Heatmap(
        z=z, x=matrix_tickers, y=matrix_tickers, colorscale="Blues", colorbar=dict(title="t-stat"),
        hoverongaps=False, hovertemplate="%{y} / %{x}<br>t=%{z:.3f}<extra></extra>",
    ))
    for a, b in matched_edge_set:
        if a in idx and b in idx:
            for r, c in [(idx[a], idx[b]), (idx[b], idx[a])]:
                matrix_fig.add_shape(type="rect", x0=c - 0.5, x1=c + 0.5, y0=r - 0.5, y1=r + 0.5,
                                      line=dict(color=(THEME["pos"] if "pos" in THEME else "#00C853"), width=3))
    matrix_fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"],
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(tickfont=dict(size=9, color=THEME["text_dim"])), yaxis=dict(tickfont=dict(size=9, color=THEME["text_dim"])),
    )
    return net_fig, matrix_fig


# ---------------------------------------------------------------------------
# Pelny rozklad wzoru TPS z realnymi wartosciami (2026-09-19, na prosbe
# wlasciciela projektu -- "widze tylko suwaki, chce widziec cala formule
# z kazda wartoscia policzona"). Wspolne dla Rebalance (Tab 4) i Sandbox.
# ---------------------------------------------------------------------------

def build_tps_formula_breakdown(mu_p, rf, sigma_p, k_p, lam, nu, tps_p=None):
    """
    Pokazuje KAZDY skladnik TPSP(w) = (mu_P - Rf) / (sigma_P*exp(nu*sigma_P)*exp(lam*K_P) + eps)
    z jego aktualna, policzona wartoscia -- nie tylko suwaki parametrow.
    Dolacza tez wprost zadany stosunek sigma_P:K_P, zeby bylo widac relatywna
    skale obu skladnikow ryzyka bez recznego liczenia.
    """
    if mu_p is None or sigma_p is None or k_p is None:
        return html.Div("Brak jeszcze policzonych wartości portfela.", style={"fontSize": "11px", "color": THEME["text_dim"]})

    lam = lam if isinstance(lam, (int, float)) else 0.0
    nu = nu if isinstance(nu, (int, float)) else 0.0
    numerator = mu_p - rf
    exp_lam_k = np.exp(lam * k_p)
    exp_nu_sigma = np.exp(nu * sigma_p)
    denominator = sigma_p * exp_nu_sigma * exp_lam_k + 1e-6
    tps_computed = numerator / denominator if denominator > 0 else float("nan")

    ratio_str = "n/d (K_P≈0)"
    if k_p and abs(k_p) > 1e-9:
        ratio = sigma_p / k_p
        ratio_str = f"{ratio:.2f} : 1" if ratio >= 1 else f"1 : {1/ratio:.2f}"

    def _row(label, value, note=""):
        return html.Div(style={"display": "flex", "justifyContent": "space-between", "padding": "4px 0", "borderBottom": f"1px solid {THEME['border']}"}, children=[
            html.Span(label, style={"fontSize": "11px", "color": THEME["text_dim"]}),
            html.Span([value, html.Span(f"  {note}", style={"color": THEME["text_dim"], "fontSize": "9.5px"}) if note else None],
                      style={"fontSize": "11.5px", "color": THEME["text_white"], "fontFamily": "monospace"}),
        ])

    return html.Div(style={"padding": "16px", "backgroundColor": THEME["bg_card"], "borderRadius": "4px", "border": f"1px solid {THEME['border_strong']}"}, children=[
        html.Div("PEŁNY ROZKŁAD WZORU TPS (wartości na dziś, przy aktualnych suwakach)", style={"fontSize": "10px", "fontWeight": "bold", "color": THEME["text_dim"], "letterSpacing": "0.5px", "marginBottom": "10px"}),
        _row("μ_P (oczekiwany zwrot portfela)", f"{mu_p*100:+.2f}%"),
        _row("R_f (stopa wolna od ryzyka / hurdle)", f"{rf*100:.2f}%"),
        _row("Licznik = μ_P − R_f", f"{numerator*100:+.2f} p.p."),
        html.Div(style={"height": "8px"}),
        _row("σ_P (√wᵀΣ_εw, zmienność Estrady)", f"{sigma_p*100:.3f}%"),
        _row("K_P (√wᵀKw, crash-overlap)", f"{k_p*100:.3f}%"),
        _row("Stosunek σ_P : K_P", ratio_str, "— relatywna skala obu ryzyk"),
        html.Div(style={"height": "8px"}),
        _row("λ (lambda, dziś)", f"{lam:.3f}"),
        _row("exp(λ·K_P)", f"{exp_lam_k:.4f}", "mnożnik kary za współkrach"),
        _row("ν (nu, dziś)", f"{nu:.3f}"),
        _row("exp(ν·σ_P)", f"{exp_nu_sigma:.4f}", "mnożnik czułości na zmienność" + (" (neutralny, ν=0)" if nu == 0 else "")),
        html.Div(style={"height": "8px"}),
        _row("Mianownik = σ_P·exp(ν·σ_P)·exp(λ·K_P) + ε", f"{denominator*100:.4f} (×10⁻²)"),
        html.Div(style={"height": "8px", "borderBottom": f"2px solid {THEME['border_strong']}"}),
        _row("TPS_P = Licznik / Mianownik", f"{tps_computed:.4f}",
             ("" if tps_p is None or abs(tps_computed - tps_p) < 1e-6 else f"(solver zwrócił {tps_p:.4f} -- sprawdź spójność)")),
    ])