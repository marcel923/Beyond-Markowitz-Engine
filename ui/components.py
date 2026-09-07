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