"""
ui/layout.py
=============
Assembles app.layout -- the full 5-tab tree (unchanged from the pre-refactor
monolith) plus the shared inspector-sidebar toggle callback.

Importing this module has the SIDE EFFECT of setting `app.layout`; app.py
imports it after all 5 tab modules (so every component id referenced by a
callback already has its callback registered) and imports it last among the
ui.* modules for the same reason -- order doesn't actually matter for
`app.layout` itself, but keeping it last is the clearest reading order.

Moved out of quant_terminal.py (Etap 0 architecture split, PROJECT_CONTEXT.md)
with NO behavior change.
"""
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State

from ui.app_instance import app
from ui.theme import THEME, TAB_STYLE, TAB_SELECTED_STYLE, TABS_CONTAINER_STYLE
from ui.components import STAGE4A_PARAMS_CONFIG, build_param_card, STAGE3_COLUMNS

app.layout = html.Div(style={
    "backgroundColor": THEME["bg_base"], "fontFamily": THEME["font"], "color": THEME["text_white"],
    "padding": "40px 50px", "minHeight": "100vh", "boxSizing": "border-box", "position": "relative"
}, children=[

    # --- WSZYSTKIE dcc.Store (globalne, poza zakładkami — dostępne niezależnie od aktywnej zakładki) ---
    dcc.Store(id="store-raw-close"),
    dcc.Store(id="store-monthly-returns"),
    dcc.Store(id="store-daily-returns"),
    dcc.Store(id="store-semicov-matrix"),
    dcc.Store(id="store-dist-matrix"),
    dcc.Store(id="store-stage3-table"),
    dcc.Store(id="store-stage3-manual-clusters"),
    dcc.Store(id="store-stage3-last-suggestion"),
    dcc.Store(id="store-stage3-final-payload"),
    dcc.Store(id="store-stage4a-tailrisk"),
    dcc.Store(id="store-crash-matrices"),
    dcc.Store(id="store-stage4a-params"),
    dcc.Store(id="store-stage4b-results"),
    dcc.Store(id="store-snapshots-refresh", data=0),

    # --- INSPEKTOR DANYCH (OFFCANVAS, globalny — dostępny z każdej zakładki) ---
    html.Div(id="data-inspector-sidebar", className="data-inspector-panel", children=[
        html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "30px"}, children=[
            html.Div([
                html.Div("SYSTEM CORE // QUANT AUDIT", style={"fontSize": "11px", "color": THEME["text_dim"], "fontWeight": "bold"}),
                html.H3("Raw Computations Inspector", style={"fontSize": "24px", "fontWeight": "700", "margin": "4px 0 0 0"})
            ]),
            html.Button("CLOSE", id="btn-inspector-close", n_clicks=0, style={
                "backgroundColor": "transparent", "color": THEME["orange"], "border": f"1px solid {THEME['border']}",
                "padding": "8px 16px", "borderRadius": "8px", "cursor": "pointer", "fontSize": "12px", "fontWeight": "bold"
            })
        ]),

        html.Label("Select Dataset to Audit:", style={"fontSize": "13px", "fontWeight": "bold", "marginBottom": "10px", "display": "block"}),
        dcc.Dropdown(
            id="dropdown-inspector-dataset",
            options=[
                {"label": "1. Surowe ceny zamknięcia (Raw Close Prices)", "value": "raw_close"},
                {"label": "2. Logarytmiczne zwroty miesięczne (Monthly Log Returns)", "value": "monthly_returns"},
                {"label": "3. Logarytmiczne zwroty dzienne (Daily Log Returns)", "value": "daily_returns"},
                {"label": "4. Macierz semi-kowariancji (Downside Semi-Covariance)", "value": "semicov_matrix"},
                {"label": "5. Macierz odległości klastrowania (Distance Matrix)", "value": "dist_matrix"}
            ],
            value="raw_close", clearable=False, style={"marginBottom": "25px"}
        ),
        html.Div(id="inspector-table-container")
    ]),

    # --- TOP BAR (globalny nagłówek, poza zakładkami) ---
    html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "30px"}, children=[
        html.Div(children=[
            html.Span("Way Stars ", style={"fontWeight": "bold", "fontSize": "16px"}),
            html.Span("®", style={"fontSize": "10px", "color": THEME["text_dim"]})
        ]),
        html.Div(style={"display": "flex", "gap": "20px", "alignItems": "center"}, children=[
            html.Button("[ OPEN RAW DATA INSPECTOR // ]", id="btn-inspector-open", n_clicks=0, style={
                "backgroundColor": "rgba(111, 44, 255, 0.15)", "color": THEME["purple"], "border": f"1px solid {THEME['purple']}",
                "padding": "10px 20px", "borderRadius": "12px", "fontWeight": "bold", "fontSize": "12px", "cursor": "pointer"
            }),
            html.Div("QUANT ENGINE ACTIVE", style={"fontSize": "12px", "color": THEME["text_dim"], "fontWeight": "500"})
        ])
    ]),

    dcc.Tabs(id="main-tabs", value="tab-1", style=TABS_CONTAINER_STYLE, children=[

        # ================= TAB 1: MARKET DATA & CLUSTERING =================
        dcc.Tab(label="1. MARKET DATA & CLUSTERING", value="tab-1", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
            html.Div(style={"paddingTop": "30px"}, children=[

                # ETAP 1 — Data Stream Ingestion
                html.Div(style={"marginBottom": "35px"}, children=[
                    html.Div(style={"backgroundColor": THEME["bg_card"], "borderRadius": "24px", "border": f"1px solid {THEME['border']}", "padding": "40px"}, children=[
                        html.Div("STAGE 01 // DATA STREAM INGESTION", style={"fontSize": "11px", "color": THEME["text_dim"], "fontWeight": "bold", "marginBottom": "10px"}),
                        html.H2("Asset Input Stream", style={"fontSize": "32px", "fontWeight": "700", "margin": "0 0 25px 0"}),
                        dcc.Textarea(
                            id="input-tickers-raw", value="NVDA, MU, AVGO, PODD, AVAV, LLY, ARGX, KKR",
                            style={"width": "100%", "height": "110px", "padding": "18px 24px", "backgroundColor": THEME["bg_input"], "border": f"1px solid {THEME['border']}", "borderRadius": "16px", "color": THEME["text_white"], "fontSize": "16px"}
                        ),
                        html.Button("INITIALIZE DATASETS →", id="btn-validate", n_clicks=0, style={"width": "100%", "marginTop": "20px", "padding": "16px", "backgroundColor": "#1C1C24", "color": "#FFFFFF", "border": "1px solid #FFFFFF", "borderRadius": "16px", "fontWeight": "600", "cursor": "pointer"}),
                        html.Div(id="validated-tags-container", style={"marginTop": "20px", "display": "flex", "flexWrap": "wrap", "gap": "10px"})
                    ])
                ]),

                # Wykres cenowy
                html.Div(id="panel-preview-container", style={"display": "none", "marginBottom": "35px"}, children=[
                    html.Div(style={"backgroundColor": THEME["bg_card"], "borderRadius": "24px", "border": f"1px solid {THEME['border']}", "padding": "40px"}, children=[
                        html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "25px"}, children=[
                            html.H3("Historical Performance Analytics", style={"fontSize": "26px", "fontWeight": "700", "margin": 0}),
                            html.Div(style={"display": "flex", "gap": "25px", "alignItems": "center"}, children=[
                                dcc.Checklist(id="checkbox-overlay-mode", options=[{"label": " OVERLAY MODE", "value": "OVERLAY"}], value=[], style={"fontSize": "13px", "color": THEME["text_white"]}),
                                html.Div(style={"width": "250px"}, children=[dcc.Dropdown(id="dropdown-active-asset", clearable=False, options=[])]),
                                dcc.Dropdown(id="dropdown-chart-timeframe", clearable=False, style={"width": "110px"}, value="5Y", options=[
                                    {"label": "5Y", "value": "5Y"}, {"label": "2Y", "value": "2Y"}, {"label": "1Y", "value": "1Y"},
                                    {"label": "6M", "value": "6M"}, {"label": "3M", "value": "3M"}, {"label": "1M", "value": "1M"}
                                ]),
                                dcc.RadioItems(id="radio-chart-type", options=[{"label": "LINEAR", "value": "linear"}, {"label": "LOG", "value": "log"}, {"label": "% RETURN", "value": "pct"}], value="linear", labelStyle={"display": "inline-block", "marginLeft": "15px", "fontSize": "12px", "color": THEME["text_dim"]})
                            ])
                        ]),
                        dcc.Graph(id="graph-asset-preview")
                    ])
                ]),

                # Tabela korelacji
                html.Div(id="panel-matrix-container", style={"display": "none", "marginBottom": "35px"}, children=[
                    html.Div(style={"backgroundColor": THEME["bg_card"], "borderRadius": "24px", "border": f"1px solid {THEME['border']}", "padding": "40px"}, children=[
                        html.H3("Asset Correlations (Monthly Returns)", style={"fontSize": "28px", "fontWeight": "700", "margin": 0}),
                        html.Div(id="matrix-date-range-sub", style={"fontSize": "12px", "color": THEME["text_dim"], "marginBottom": "25px"}),
                        html.Div(id="table-correlation-wrapper")
                    ])
                ]),

                # ETAP 2 — konfiguracja klastrowania
                html.Div(id="panel-config-stage2-container", style={"display": "none", "marginBottom": "35px"}, children=[
                    html.Div(style={"backgroundColor": THEME["bg_card"], "borderRadius": "24px", "border": f"1px solid {THEME['border']}", "padding": "40px"}, children=[
                        html.Div("STAGE 02 // ALGORITHMIC CONFIGURATION", style={"fontSize": "11px", "color": THEME["text_dim"], "fontWeight": "bold", "marginBottom": "10px"}),
                        html.H2("Engine Configuration", style={"fontSize": "32px", "fontWeight": "700", "margin": "0 0 25px 0"}),
                        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "25px"}, children=[
                            html.Div([
                                html.Label("Clustering Method (dane dobierane automatycznie):", style={"fontSize": "13px", "color": THEME["text_white"], "display": "block", "marginBottom": "8px"}),
                                dcc.Dropdown(id="dropdown-cluster-method", options=[
                                    {"label": "Ward Linkage — Standard Correlation (Monthly)", "value": "ward"},
                                    {"label": "Complete Linkage — Standard Correlation (Monthly)", "value": "complete"},
                                    {"label": "Single Linkage — Standard Correlation (Monthly)", "value": "single"},
                                    {"label": "Average Linkage — Standard Correlation (Monthly)", "value": "average"},
                                    {"label": "Semi-Covariance / Downside Risk (Daily, Ward)", "value": "semicov"},
                                    {"label": "RMT Denoised Matrix — Marchenko-Pastur (Daily, Ward)", "value": "rmt_denoised"},
                                    {"label": "DBSCAN — Density-Based (Monthly)", "value": "dbscan"},
                                    {"label": "Time-Series DTW + K-Medoids (Daily Price Shape)", "value": "dtw_kmeans"}
                                ], value="ward", clearable=False)
                            ]),
                            html.Div([
                                html.Label(id="label-k-count", children="Target Cluster Count (k):", style={"fontSize": "13px", "color": THEME["text_white"], "display": "block", "marginBottom": "8px"}),
                                dcc.Dropdown(id="dropdown-k-count", options=[{"label": str(i), "value": i} for i in range(2, 7)], value=3, clearable=False),
                                html.Button("SUGGEST OPTIMAL K (SILHOUETTE)", id="btn-suggest-k", n_clicks=0, style={
                                    "marginTop": "10px", "width": "100%", "padding": "10px", "backgroundColor": "transparent",
                                    "color": THEME["purple"], "border": f"1px solid {THEME['purple']}", "borderRadius": "10px",
                                    "fontSize": "12px", "fontWeight": "bold", "cursor": "pointer"
                                }),
                                html.Div(id="suggested-k-output", style={"fontSize": "12px", "color": THEME["text_dim"], "marginTop": "8px"})
                            ])
                        ]),
                        html.Button("EXECUTE CLUSTERING PIPELINE →", id="btn-cluster-execute", n_clicks=0, style={"width": "100%", "padding": "16px", "backgroundColor": "#FFFFFF", "color": "#0B0B0E", "border": "none", "borderRadius": "16px", "fontWeight": "700", "cursor": "pointer"})
                    ])
                ]),

                # Widget klastracji (dendrogram + heatmap)
                html.Div(id="panel-clustering-outputs-container", style={"display": "none", "marginBottom": "35px"}, children=[
                    html.Div(style={"backgroundColor": THEME["bg_card"], "borderRadius": "24px", "border": f"1px solid {THEME['border']}", "padding": "40px"}, children=[
                        html.Div("MACHINE LEARNING ENGINE OUTPUT", style={"fontSize": "11px", "color": THEME["text_dim"], "fontWeight": "bold", "marginBottom": "5px"}),
                        html.H3("Hierarchical Agglomerative Analytics", style={"fontSize": "26px", "fontWeight": "700", "marginBottom": "20px"}),
                        html.Div(id="cluster-legend-output", style={"padding": "20px", "backgroundColor": THEME["bg_input"], "borderRadius": "16px", "marginBottom": "30px", "border": f"1px solid {THEME['border']}"}),

                        html.Div(style={"display": "flex", "flexDirection": "column", "gap": "35px"}, children=[
                            html.Div([
                                html.Div("Clustered Dendrogram Tree (Grouped Color-Coding)", style={"fontSize": "14px", "fontWeight": "bold", "marginBottom": "15px"}),
                                dcc.Graph(id="graph-dendrogram")
                            ]),
                            html.Div(style={"display": "flex", "flexDirection": "column", "alignItems": "center"}, children=[
                                html.Div("Permutated Clustered Heatmap Matrix", style={"fontSize": "14px", "fontWeight": "bold", "width": "450px", "marginBottom": "15px"}),
                                dcc.Graph(id="graph-clustered-heatmap", style={"width": "450px", "height": "450px"})
                            ])
                        ])
                    ])
                ]),
                html.Div(id="error-output", style={"marginTop": "30px", "color": THEME["orange"], "fontSize": "14px", "fontWeight": "bold"}),
            ])
        ]),

        # ================= TAB 2: FUNDAMENTAL INPUTS =================
        dcc.Tab(label="2. FUNDAMENTAL INPUTS", value="tab-2", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
            html.Div(style={"paddingTop": "30px"}, children=[
                html.Div(id="panel-stage3-container", style={"display": "none", "marginBottom": "35px"}, children=[
                    html.Div(style={"backgroundColor": THEME["bg_card"], "borderRadius": "24px", "border": f"1px solid {THEME['border']}", "padding": "40px"}, children=[
                        html.Div("STAGE 03 // PORTFOLIO CONSTRUCTION INPUTS", style={"fontSize": "11px", "color": THEME["text_dim"], "fontWeight": "bold", "marginBottom": "10px"}),
                        html.H2("Asset Metadata & Cluster Assignment", style={"fontSize": "32px", "fontWeight": "700", "margin": "0 0 25px 0"}),

                        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1.4fr", "gap": "25px", "marginBottom": "25px", "alignItems": "start"}, children=[
                            html.Div([
                                html.Label("Baseline Clustering Model (initial suggestion source):", style={"fontSize": "13px", "color": THEME["text_white"], "display": "block", "marginBottom": "8px"}),
                                dcc.Dropdown(id="dropdown-baseline-model", clearable=False, value="semicov_ward", options=[
                                    {"label": "Semi-Variance + Ward Linkage", "value": "semicov_ward"},
                                    {"label": "Semi-Variance + Single Linkage", "value": "semicov_single"},
                                    {"label": "Semi-Variance + Complete Linkage", "value": "semicov_complete"},
                                    {"label": "Standard Variance + Ward Linkage", "value": "std_ward"},
                                    {"label": "Standard Variance + Single Linkage", "value": "std_single"},
                                    {"label": "Standard Variance + Complete Linkage", "value": "std_complete"},
                                ]),
                                html.Div("Zmiana modelu przelicza sugestie klastrów tylko dla wierszy, których ręcznie nie nadpisałeś. Dane fundamentalne (kolumny 3-9) nigdy nie są ruszane.",
                                         style={"fontSize": "11px", "color": THEME["text_dim"], "marginTop": "10px", "lineHeight": "1.5"}),
                                html.Button("↺ Reset ręcznych nadpisań klastrów", id="btn-stage3-reset-overrides", n_clicks=0, style={
                                    "marginTop": "14px", "width": "100%", "padding": "10px", "backgroundColor": "transparent",
                                    "color": THEME["text_dim"], "border": f"1px solid {THEME['border']}", "borderRadius": "10px",
                                    "fontSize": "11px", "cursor": "pointer"
                                })
                            ]),
                            html.Div(id="stage3-summary-panel", style={"padding": "20px", "backgroundColor": THEME["bg_input"], "borderRadius": "16px", "border": f"1px solid {THEME['border']}", "minHeight": "140px"})
                        ]),

                        dash_table.DataTable(
                            id="table-stage3-assets", columns=STAGE3_COLUMNS, data=[], editable=True,
                            page_action='native', page_size=20, row_deletable=False,
                            style_header={'backgroundColor': '#0B0B0E', 'color': THEME['text_white'], 'fontWeight': 'bold', 'border': '1px solid #222230', 'padding': '10px', 'fontSize': '12px'},
                            style_data={'backgroundColor': THEME['bg_input'], 'color': THEME['text_white'], 'border': '1px solid #222230', 'fontFamily': THEME['font']},
                            style_cell={'padding': '10px', 'textAlign': 'center', 'fontSize': '13px'},
                            style_cell_conditional=[{'if': {'column_id': 'Ticker'}, 'fontWeight': 'bold', 'textAlign': 'left', 'color': THEME['purple']}],
                            style_data_conditional=[{'if': {'column_id': 'Assigned Cluster'}, 'backgroundColor': '#1C1C24', 'fontWeight': 'bold', 'color': THEME['orange']}]
                        ),

                        html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginTop": "25px"}, children=[
                            html.Div(id="stage3-confirm-output", style={"fontSize": "13px", "color": THEME["text_dim"]}),
                            html.Button("CONFIRM & EXPORT TO STAGE 4 SOLVER", id="btn-stage3-confirm", n_clicks=0, style={
                                "padding": "16px 28px", "backgroundColor": "#FFFFFF", "color": "#0B0B0E", "border": "none",
                                "borderRadius": "16px", "fontWeight": "700", "cursor": "pointer", "fontSize": "13px"
                            })
                        ])
                    ])
                ])
            ])
        ]),

        # ================= TAB 3: TAIL-RISK & UNDERWATER ANALYTICS =================
        dcc.Tab(label="3. TAIL-RISK & UNDERWATER ANALYTICS", value="tab-3", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
            html.Div(style={"paddingTop": "30px"}, children=[
                html.Div(id="panel-stage4a-tailrisk-container", style={"display": "none", "marginBottom": "35px"}, children=[
                    html.Div(style={"backgroundColor": THEME["bg_card"], "borderRadius": "24px", "border": f"1px solid {THEME['border']}", "padding": "40px"}, children=[
                        html.Div("SECTION 4.1 // TAIL-RISK ANALYTICS ENGINE", style={"fontSize": "11px", "color": THEME["text_dim"], "fontWeight": "bold", "marginBottom": "10px"}),
                        html.H2("Empirical Quantile Drawdown (CDD 0.10)", style={"fontSize": "26px", "fontWeight": "700", "margin": "0 0 25px 0"}),

                        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1.3fr", "gap": "25px", "alignItems": "start"}, children=[
                            html.Div([
                                html.Div("TAIL-RISK SUMMARY GRID", style={"fontSize": "12px", "color": THEME["text_dim"], "fontWeight": "bold", "marginBottom": "12px"}),
                                html.Div(id="stage4a-summary-table")
                            ]),
                            html.Div([
                                html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "12px"}, children=[
                                    html.Div("UNDERWATER DRAWDOWN INSPECTOR", style={"fontSize": "12px", "color": THEME["text_dim"], "fontWeight": "bold"}),
                                    html.Div(style={"width": "160px"}, children=[dcc.Dropdown(id="dropdown-stage4a-stock", clearable=False, placeholder="Select Stock")])
                                ]),
                                dcc.Graph(id="graph-stage4a-underwater", config={"displayModeBar": False})
                            ])
                        ])
                    ])
                ]),

                # --- Sekcja 3.3: Discrete Crash-Overlap Matrix (K = J ⊙ S) ---
                html.Div(id="panel-crash-overlap-container", style={"display": "none", "marginBottom": "35px"}, children=[
                    html.Div(style={"backgroundColor": THEME["bg_card"], "borderRadius": "24px", "border": f"1px solid {THEME['border']}", "padding": "40px"}, children=[
                        html.Div("SECTION 3.3 // DISCRETE CRASH-OVERLAP MATRIX", style={"fontSize": "11px", "color": THEME["text_dim"], "fontWeight": "bold", "marginBottom": "10px"}),
                        html.H2("Crash-Risk Coupling: K = J ⊙ S", style={"fontSize": "26px", "fontWeight": "700", "margin": "0 0 6px 0"}),
                        html.Div("Pary liczone na wspólnych sesjach handlowych (przecięcie dat per para) — spółki o różnych giełdach/kalendarzach lub krótszą historią nie zaniżają sobie nawzajem pokrycia.",
                                 style={"fontSize": "12px", "color": THEME["text_dim"], "marginBottom": "25px", "maxWidth": "780px", "lineHeight": "1.5"}),

                        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "25px", "alignItems": "start", "marginBottom": "30px"}, children=[
                            html.Div([
                                html.Div("TEMPORAL OVERLAP — MATRIX J (Jaccard, 0–1)", style={"fontSize": "12px", "color": THEME["text_dim"], "fontWeight": "bold", "marginBottom": "12px"}),
                                dcc.Graph(id="graph-crash-jaccard", config={"displayModeBar": False})
                            ]),
                            html.Div([
                                html.Div("COMPOSITE CRASH-RISK — MATRIX K = J ⊙ S", style={"fontSize": "12px", "color": THEME["text_dim"], "fontWeight": "bold", "marginBottom": "12px"}),
                                dcc.Graph(id="graph-crash-k", config={"displayModeBar": False})
                            ])
                        ]),

                        html.Div("CRASH-RISK CONTRIBUTION RANKING (row-sum of K, per spółka)", style={"fontSize": "12px", "color": THEME["text_dim"], "fontWeight": "bold", "marginBottom": "12px"}),
                        html.Div(id="crash-ranking-table")
                    ])
                ])
            ])
        ]),

        # ================= TAB 4: STRATEGY PARAMS, TPS & 3D OPTIMIZATION =================
        dcc.Tab(label="4. STRATEGY PARAMS, TPS & 3D OPTIMIZATION", value="tab-4", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
            html.Div(style={"paddingTop": "30px"}, children=[

                html.Div(id="panel-stage4a-container", style={"display": "none", "marginBottom": "35px"}, children=[
                    html.Div(style={"backgroundColor": THEME["bg_card"], "borderRadius": "24px", "border": f"1px solid {THEME['border']}", "padding": "40px"}, children=[
                        html.Div("STAGE 04A // GLOBAL STRATEGY PARAMETERS", style={"fontSize": "11px", "color": THEME["text_dim"], "fontWeight": "bold", "marginBottom": "10px"}),
                        html.H2("Strategy Control Parameters", style={"fontSize": "32px", "fontWeight": "700", "margin": "0 0 25px 0"}),
                        html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "16px"},
                                 children=[build_param_card(cfg) for cfg in STAGE4A_PARAMS_CONFIG]),
                    ])
                ]),

                html.Div(id="panel-stage4b-container", style={"display": "none", "marginBottom": "35px", "marginTop": "35px"}, children=[
                    html.Div(style={"backgroundColor": THEME["bg_card"], "borderRadius": "24px", "border": f"1px solid {THEME['border']}", "padding": "40px"}, children=[
                        html.Div("STAGE 04B // TPS METRIC & TWO-STAGE SLSQP PORTFOLIO SOLVER", style={"fontSize": "11px", "color": THEME["text_dim"], "fontWeight": "bold", "marginBottom": "10px"}),
                        html.H2("Optimization Results & Allocation Panel", style={"fontSize": "32px", "fontWeight": "700", "margin": "0 0 25px 0"}),

                        html.Div(id="stage4b-error-banner"),
                        html.Div(id="stage4b-kpi-row", style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "16px", "marginBottom": "35px"}),

                        html.Hr(style={"border": "none", "borderTop": f"1px solid {THEME['border']}", "margin": "10px 0 30px 0"}),

                        html.Div("ALLOCATION BREAKDOWN", style={"fontSize": "12px", "color": THEME["text_dim"], "fontWeight": "bold", "marginBottom": "12px"}),
                        html.Div(style={"display": "grid", "gridTemplateColumns": "0.9fr 1.4fr", "gap": "25px", "alignItems": "start", "marginBottom": "35px"}, children=[
                            dcc.Graph(id="graph-stage4b-donut", config={"displayModeBar": False}),
                            dash_table.DataTable(
                                id="table-stage4b-weights", data=[], page_action='native', page_size=15, sort_action='native',
                                columns=[
                                    {"name": "Ticker", "id": "Ticker"},
                                    {"name": "Cluster", "id": "Cluster"},
                                    {"name": "Intra-Cluster Weight (g_i)", "id": "g_i", "type": "numeric", "format": {"specifier": ".1%"}},
                                    {"name": "Final Portfolio Weight", "id": "w_i", "type": "numeric", "format": {"specifier": ".2%"}},
                                    {"name": "Standalone Asset TPS (diagnostyka, nie steruje alokacją)", "id": "TPS_i", "type": "numeric", "format": {"specifier": "+.2f"}},
                                ],
                                style_header={'backgroundColor': '#0B0B0E', 'color': THEME['text_white'], 'fontWeight': 'bold', 'border': '1px solid #222230', 'padding': '10px', 'fontSize': '12px'},
                                style_data={'backgroundColor': THEME['bg_input'], 'color': THEME['text_white'], 'border': '1px solid #222230'},
                                style_cell={'padding': '10px', 'textAlign': 'center', 'fontSize': '13px'},
                                style_cell_conditional=[{'if': {'column_id': 'Ticker'}, 'fontWeight': 'bold', 'textAlign': 'left', 'color': THEME['purple']}],
                                style_data_conditional=[
                                    {'if': {'filter_query': '{w_i} = 0', 'column_id': 'w_i'}, 'color': THEME['text_dim']},
                                    {'if': {'filter_query': '{Cluster} contains "★"', 'column_id': 'Cluster'}, 'color': THEME['orange'], 'fontWeight': 'bold'},
                                ]
                            )
                        ]),

                        html.Div(id="panel-singleton-split", style={"display": "none", "marginBottom": "30px"}),

                        html.Hr(style={"border": "none", "borderTop": f"1px solid {THEME['border']}", "margin": "10px 0 30px 0"}),

                        html.Div("3D EFFICIENT RISK-REWARD SURFACE", style={"fontSize": "12px", "color": THEME["text_dim"], "fontWeight": "bold", "marginBottom": "12px"}),
                        dcc.Graph(id="graph-stage4b-3d", config={"displayModeBar": False}),
                    ])
                ])
            ])
        ]),

        # ================= TAB 5: FORWARD TRACKER =================
        dcc.Tab(label="5. FORWARD TRACKER", value="tab-5", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
            html.Div(style={"paddingTop": "30px"}, children=[
                html.Div(id="panel-stage4b-tracker-container", style={"display": "none", "marginBottom": "35px"}, children=[
                    html.Div(style={"backgroundColor": THEME["bg_card"], "borderRadius": "24px", "border": f"1px solid {THEME['border']}", "padding": "40px"}, children=[

                        html.Div("FORWARD EVALUATION & HYPERPARAMETER SANDBOX", style={"fontSize": "11px", "color": THEME["purple"], "fontWeight": "bold", "letterSpacing": "1.5px", "marginBottom": "8px"}),
                        html.H2("Quantitative Forward-Testing Terminal", style={"fontSize": "30px", "fontWeight": "700", "margin": "0 0 30px 0"}),

                        # --- TOP BAR: wybór snapshotu + metadane + live sync + zapis ---
                        html.Div(style={"display": "grid", "gridTemplateColumns": "1.4fr 1fr", "gap": "20px", "marginBottom": "30px", "alignItems": "stretch"}, children=[
                            html.Div(style={"padding": "22px", "backgroundColor": "#161B22", "borderRadius": "16px", "border": "1px solid #30363D"}, children=[
                                html.Div("ACTIVE SNAPSHOT", style={"fontSize": "10px", "fontWeight": "bold", "color": THEME["text_dim"], "letterSpacing": "1px", "marginBottom": "10px"}),
                                dcc.Dropdown(id="dropdown-snapshot-select", options=[], placeholder="Wybierz zapisany portfel...",
                                             clearable=False, className="dark-dropdown", style={"marginBottom": "10px"}),
                                html.Div(id="snapshot-meta-info", style={"fontSize": "11px", "color": THEME["text_dim"], "marginBottom": "16px"}),
                                html.Div(style={"display": "flex", "gap": "8px"}, children=[
                                    html.Button("REFRESH LIVE DATA", id="btn-refresh-live", n_clicks=0, style={
                                        "flex": "2", "padding": "12px", "backgroundColor": THEME["purple"], "color": "#FFFFFF",
                                        "border": "none", "borderRadius": "10px", "fontWeight": "700", "cursor": "pointer",
                                        "fontSize": "11px", "letterSpacing": "0.5px"
                                    }),
                                    dcc.Input(id="input-rename-snapshot", type="text", placeholder="Nowa nazwa...", style={
                                        "flex": "2", "padding": "0 10px", "borderRadius": "8px", "border": "1px solid #30363D",
                                        "backgroundColor": "#0D1117", "color": "#FFFFFF", "fontSize": "11px", "boxSizing": "border-box"
                                    }),
                                    html.Button("RENAME", id="btn-rename-snapshot", n_clicks=0, title="Zmień nazwę", style={
                                        "flex": "0 0 auto", "padding": "0 14px", "backgroundColor": "#0D1117", "color": THEME["text_dim"],
                                        "border": "1px solid #30363D", "borderRadius": "8px", "cursor": "pointer", "fontSize": "10px", "fontWeight": "700"
                                    }),
                                    html.Button("DELETE", id="btn-delete-snapshot", n_clicks=0, title="Usuń zaznaczony snapshot", style={
                                        "flex": "0 0 auto", "padding": "0 14px", "backgroundColor": "#0D1117", "color": THEME["orange"],
                                        "border": f"1px solid {THEME['orange']}", "borderRadius": "8px", "cursor": "pointer", "fontSize": "10px", "fontWeight": "700"
                                    })
                                ]),
                                html.Div(id="delete-confirm-banner", style={"marginTop": "10px"}),
                                dcc.Store(id="store-delete-armed", data=None)
                            ]),

                            # --- ZAPIS BIEŻĄCEGO PORTFELA ---
                            html.Div(style={"padding": "22px", "backgroundColor": "#161B22", "borderRadius": "16px", "border": "1px solid #30363D", "display": "flex", "flexDirection": "column", "justifyContent": "space-between"}, children=[
                                html.Div([
                                    html.Div("SAVE CURRENT PORTFOLIO", style={"fontSize": "10px", "fontWeight": "bold", "color": THEME["text_dim"], "letterSpacing": "1px", "marginBottom": "10px"}),
                                    html.Div("Zapisuje zoptymalizowany portfel ze Stage 4B jako nowy punkt startowy do trackingu.",
                                             style={"fontSize": "10px", "color": THEME["text_dim"], "lineHeight": "1.5", "marginBottom": "14px"}),
                                    dcc.Input(id="input-snapshot-name", type="text", placeholder="np. Defensive_Lambda0.8_v1", style={
                                        "width": "100%", "padding": "10px", "borderRadius": "10px", "border": "1px solid #30363D",
                                        "backgroundColor": "#0D1117", "color": "#FFFFFF", "boxSizing": "border-box", "fontSize": "12px", "marginBottom": "12px"
                                    }),
                                ]),
                                html.Div([
                                    html.Button("SAVE PORTFOLIO", id="btn-save-snapshot", n_clicks=0, style={
                                        "width": "100%", "padding": "12px", "backgroundColor": "#FFFFFF", "color": "#0B0B0E",
                                        "border": "none", "borderRadius": "10px", "fontWeight": "700", "cursor": "pointer", "fontSize": "11px", "letterSpacing": "0.5px"
                                    }),
                                    html.Div(id="snapshot-save-status", style={"marginTop": "10px", "fontSize": "11px"})
                                ])
                            ])
                        ]),

                        # --- KPI ROW ---
                        html.Div(id="kpi-summary-row", style={"display": "grid", "gridTemplateColumns": "repeat(5, 1fr)", "gap": "12px", "marginBottom": "30px"}),

                        html.Div(style={"borderTop": f"1px solid {THEME['border']}", "marginBottom": "30px"}),

                        # --- GŁÓWNA SIATKA: SANDBOX | WYKRESY ---
                        html.Div(style={"display": "grid", "gridTemplateColumns": "360px 1fr", "gap": "20px"}, children=[

                            # LEWA KOLUMNA: PARAMETER SANDBOX
                            html.Div(style={"padding": "24px", "backgroundColor": "#161B22", "borderRadius": "16px", "border": "1px solid #30363D", "borderLeft": f"3px solid {THEME['orange']}"}, children=[
                                html.Div("HYPERPARAMETER SANDBOX", style={"fontSize": "11px", "fontWeight": "bold", "color": THEME["orange"], "letterSpacing": "1px", "marginBottom": "8px"}),
                                html.Div("Modyfikuj parametry ręcznie, aby przeliczyć wagi dokładnie tym samym silnikiem True Two-Stage SLSQP (Section 3.4) + Singleton Split, którego używa główny solver w Tab 4 — bez ponownego pobierania danych.",
                                         style={"fontSize": "10px", "color": THEME["text_dim"], "marginBottom": "22px", "lineHeight": "1.6"}),

                                html.Label("ALPHA — Growth/Upside Blend (α)", style={"fontSize": "10px", "fontWeight": "bold", "color": THEME["text_dim"], "letterSpacing": "0.5px"}),
                                dcc.Slider(id="slider-sb-alpha", min=0.0, max=1.0, step=0.05, value=0.5, marks={0.0: "0 (Upside)", 0.5: "0.5", 1.0: "1 (Growth)"}, tooltip={"placement": "bottom", "always_visible": True}),

                                html.Label("LAMBDA — kara za współkrach (λ, K = J⊙S)", style={"fontSize": "10px", "fontWeight": "bold", "color": THEME["text_dim"], "letterSpacing": "0.5px", "marginTop": "20px", "display": "block"}),
                                dcc.Slider(id="slider-sb-lambda", min=0.1, max=25.0, step=0.1, value=3.0, marks={0.1: "0.1", 5: "5", 10: "10", 15: "15", 25: "25"}, tooltip={"placement": "bottom", "always_visible": True}),

                                html.Label("GAMMA — kara za rozstrzał widełek (γ)", style={"fontSize": "10px", "fontWeight": "bold", "color": THEME["text_dim"], "letterSpacing": "0.5px", "marginTop": "20px", "display": "block"}),
                                dcc.Slider(id="slider-sb-gamma", min=0.1, max=5.0, step=0.1, value=1.5, marks={0.1: "0.1", 1: "1", 2.5: "2.5", 5: "5"}, tooltip={"placement": "bottom", "always_visible": True}),

                                html.Label("KAPPA — momentum rewizji EPS (κ)", style={"fontSize": "10px", "fontWeight": "bold", "color": THEME["text_dim"], "letterSpacing": "0.5px", "marginTop": "20px", "display": "block"}),
                                dcc.Slider(id="slider-sb-kappa", min=0.0, max=5.0, step=0.1, value=1.0, marks={0: "0", 2.5: "2.5", 5: "5"}, tooltip={"placement": "bottom", "always_visible": True}),

                                html.Label("MAX SINGLE WEIGHT (w_max)", style={"fontSize": "10px", "fontWeight": "bold", "color": THEME["text_dim"], "letterSpacing": "0.5px", "marginTop": "20px", "display": "block"}),
                                dcc.Slider(id="slider-sb-wmax", min=0.05, max=1.0, step=0.05, value=0.30, marks={0.05: "5%", 0.3: "30%", 0.6: "60%", 1.0: "100%"}, tooltip={"placement": "bottom", "always_visible": True}),

                                html.Label("RISK-FREE RATE — hurdle rate (R_f)", style={"fontSize": "10px", "fontWeight": "bold", "color": THEME["text_dim"], "letterSpacing": "0.5px", "marginTop": "20px", "display": "block"}),
                                dcc.Slider(id="slider-sb-rf", min=0.0, max=0.25, step=0.005, value=0.045, marks={0.0: "0%", 0.10: "10%", 0.25: "25%"}, tooltip={"placement": "bottom", "always_visible": True}),

                                html.Label("N_ref — próg pokrycia analityków (A_i)", style={"fontSize": "10px", "fontWeight": "bold", "color": THEME["text_dim"], "letterSpacing": "0.5px", "marginTop": "20px", "display": "block"}),
                                dcc.Slider(id="slider-sb-nref", min=3, max=30, step=1, value=8, marks={3: "3", 15: "15", 30: "30"}, tooltip={"placement": "bottom", "always_visible": True}),

                                html.Div(style={"borderTop": "1px solid #30363D", "margin": "24px 0 0 0", "paddingTop": "14px"}, children=[
                                    html.Div("SANDBOX PORTFOLIO METRICS", style={"fontSize": "10px", "fontWeight": "bold", "color": THEME["text_dim"], "letterSpacing": "0.5px", "marginBottom": "10px"}),
                                    html.Div(id="sandbox-mini-kpi-row")
                                ])
                            ]),

                            # PRAWA KOLUMNA: WYKRESY
                            html.Div(style={"padding": "24px", "backgroundColor": "#161B22", "borderRadius": "16px", "border": "1px solid #30363D", "borderLeft": f"3px solid {THEME['purple']}"}, children=[
                                html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "16px", "flexWrap": "wrap", "gap": "10px"}, children=[
                                    html.Div("FORWARD CUMULATIVE PERFORMANCE — OD DNIA ZAPISU", style={"fontSize": "11px", "fontWeight": "bold", "color": "#FFFFFF", "letterSpacing": "0.5px"}),
                                    dcc.Checklist(
                                        id="checklist-benchmarks",
                                        options=[
                                            {"label": " Equal-Weight (1/N)", "value": "1N"},
                                            {"label": " Equal Risk (Inv-Vol)", "value": "INV_VOL"},
                                            {"label": " S&P 500 (SPY)", "value": "SPY"},
                                            {"label": " Nasdaq 100 (QQQ)", "value": "QQQ"}
                                        ],
                                        value=["1N", "SPY"],
                                        labelStyle={"display": "inline-block", "marginLeft": "12px", "fontSize": "10px", "color": THEME["text_dim"]}
                                    )
                                ]),
                                dcc.Graph(id="graph-forward-equity-curves", style={"height": "420px"}, config={"displayModeBar": False}),

                                html.Div(style={"marginTop": "24px", "display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}, children=[
                                    html.Div([
                                        html.Div("STOCK-LEVEL RETURN BREAKDOWN", style={"fontSize": "10px", "fontWeight": "bold", "color": THEME["text_dim"], "letterSpacing": "0.5px", "marginBottom": "10px"}),
                                        dcc.Graph(id="graph-asset-returns-bar", style={"height": "200px"}, config={"displayModeBar": False})
                                    ]),
                                    html.Div([
                                        html.Div("UNDERWATER DRAWDOWN COMPARISON", style={"fontSize": "10px", "fontWeight": "bold", "color": THEME["text_dim"], "letterSpacing": "0.5px", "marginBottom": "10px"}),
                                        dcc.Graph(id="graph-forward-drawdowns", style={"height": "200px"}, config={"displayModeBar": False})
                                    ])
                                ])
                            ])
                        ]),

                        # --- SEKCJA PEŁNEJ SZEROKOŚCI: SZCZEGÓŁOWA TABELA WAG (Original vs Sandbox) ---
                        html.Div(style={"borderTop": f"1px solid {THEME['border']}", "margin": "30px 0"}),
                        html.Div(style={"padding": "24px", "backgroundColor": "#161B22", "borderRadius": "16px", "border": "1px solid #30363D"}, children=[
                            html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "6px", "flexWrap": "wrap", "gap": "10px"}, children=[
                                html.Div("WEIGHT BREAKDOWN: ORIGINAL vs SANDBOX (pełne dane, aktualizowane na żywo z suwakami)",
                                         style={"fontSize": "11px", "fontWeight": "bold", "color": "#FFFFFF", "letterSpacing": "0.5px"}),
                            ]),
                            html.Div("Kolumna Sandbox przelicza się natychmiast po zmianie dowolnego suwaka (λ, γ, κ, w_max, R_f, N_ref) — dokładnie tym samym silnikiem True Two-Stage SLSQP + Singleton Split co Tab 4.",
                                     style={"fontSize": "10px", "color": THEME["text_dim"], "marginBottom": "16px", "lineHeight": "1.5"}),
                            html.Div(id="sandbox-weights-table-container")
                        ])
                    ])
                ])
            ])
        ]),
    ])
])



# --- SYSTEM CALLBACKS ---