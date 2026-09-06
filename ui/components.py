"""
ui/components.py
==================
Reusable, generic UI building blocks shared across multiple tabs: color-blend
DataTable cell styling, KPI cards, and the Stage 4 parameter input card
(driven by STAGE4A_PARAMS_CONFIG).

Moved out of quant_terminal.py (Etap 0 architecture split, PROJECT_CONTEXT.md)
with NO behavior change.
"""
import numpy as np
import pandas as pd
from dash import dcc, html

from ui.theme import THEME

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return np.array([int(hex_str[i:i+2], 16) for i in (0, 2, 4)])

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def generate_tws_matrix_styles(df, tickers):
    styles = []
    c_bg, c_purp, c_oran = hex_to_rgb(THEME["bg_card"]), hex_to_rgb(THEME["purple"]), hex_to_rgb(THEME["orange"])
    for t in tickers:
        for _, row in df.iterrows():
            val = row[t]
            if pd.isna(val): continue
            rgb_mixed = c_bg + (c_purp - c_bg) * val if val >= 0 else c_bg + (c_oran - c_bg) * abs(val)
            styles.append({'if': {'filter_query': f'{{Ticker}} eq "{row["Ticker"]}"', 'column_id': t}, 'backgroundColor': rgb_to_hex(np.clip(rgb_mixed, 0, 255)), 'color': '#FFFFFF'})
    return styles

STAGE4A_PARAMS_CONFIG = [
    {"id": "alpha", "name": "Alpha Blend", "symbol": "α", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
     "comment": "Growth/Upside Blend (Section 4.2, Etap 2). α=0.0 → pure analyst target-price consensus. α=0.5 → equal blend. α=1.0 → pure fundamental EPS growth (the analyst dispersion penalty γ drops out entirely at α=1.0, since it only ever discounts the target-price branch). Analyst coverage confidence A(N_i) still discounts the result at any α — coverage depth is treated as a general forecast-quality signal, not specific to target prices."},
    {"id": "lambda", "name": "Lambda", "symbol": "λ", "default": 3.0, "min": 0.1, "max": 25.0, "step": 0.1,
     "comment": "Crash-overlap penalty sensitivity: exp(λ·√(wᵀKw)). Rescaled for the K-based quadratic penalty (Section 3.3/3.4) — √(wᵀKw) typically runs ≈0.05–0.20, so λ needs a much wider range than the old per-asset Z-score exponent did to have a comparable effect."},
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
        "backgroundColor": THEME["bg_input"], "border": f"1px solid {THEME['border']}", "borderRadius": "16px",
        "padding": "18px", "flex": "1 1 220px", "minWidth": "220px"
    }, children=[
        html.Div([
            html.Span(cfg["name"], style={"fontSize": "13px", "fontWeight": "bold", "color": THEME["text_white"]}),
            html.Span(f"  {cfg['symbol']}", style={"fontSize": "13px", "color": THEME["purple"], "fontWeight": "bold", "marginLeft": "4px"})
        ], style={"marginBottom": "10px"}),
        dcc.Input(id=f"input-{cfg['id']}", type="number", value=cfg["default"], step=step,
                   style={"width": "100%", "padding": "10px", "backgroundColor": THEME["bg_base"], "border": f"1px solid {THEME['border']}",
                          "borderRadius": "8px", "color": THEME["text_white"], "fontSize": "15px", "fontWeight": "bold", "boxSizing": "border-box"}),
        html.Div(id=f"badge-{cfg['id']}", children=f"[Recommended: {cfg['min']} – {cfg['max']}]", style={
            "fontSize": "10px", "color": THEME["text_dim"], "marginTop": "8px", "padding": "3px 8px",
            "border": f"1px solid {THEME['border']}", "borderRadius": "20px", "display": "inline-block"
        }),
        html.Div(cfg["comment"], style={"fontSize": "11px", "color": THEME["text_dim"], "marginTop": "10px", "lineHeight": "1.5"})
    ])

def build_kpi_card(label, value_str, sub_str="", color=None):
    color = color or THEME["text_white"]
    return html.Div(style={
        "backgroundColor": THEME["bg_input"], "border": f"1px solid {THEME['border']}", "borderRadius": "16px", "padding": "20px"
    }, children=[
        html.Div(label, style={"fontSize": "11px", "color": THEME["text_dim"], "fontWeight": "bold", "marginBottom": "10px", "letterSpacing": "0.5px"}),
        html.Div(value_str, style={"fontSize": "28px", "fontWeight": "700", "color": color}),
        html.Div(sub_str, style={"fontSize": "11px", "color": THEME["text_dim"], "marginTop": "6px"})
    ])


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