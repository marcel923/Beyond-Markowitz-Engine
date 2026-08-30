import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import base64
import traceback
from datetime import datetime, timedelta
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram, fcluster
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
import tps_solver as ts
import snapshot_store as snap

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# --- BRUTALNY ODPORNY CIEMNY CSS (DATA URI B64) ---
dark_css = """
.Select-control, .Select, .Select-multi-value-wrapper {
    background-color: #1C1C24 !important;
    color: #FFFFFF !important;
    border: 1px solid #222230 !important;
    border-radius: 12px !important;
    height: 38px !important;
}
.Select-placeholder {
    color: #8A8A98 !important;
    line-height: 38px !important;
}
.Select-value-label, .Select-value {
    color: #FFFFFF !important;
    background-color: #2F2F3E !important;
    border: 1px solid #3F3F52 !important;
    border-radius: 6px !important;
    line-height: 28px !important;
}
.Select--multi .Select-value {
    background-color: #13131A !important;
    border: 1px solid #6F2CFF !important;
}
.Select-menu-outer, .Select-menu, .VirtualizedSelectOption {
    background-color: #1C1C24 !important;
    color: #FFFFFF !important;
    border: 1px solid #222230 !important;
}
.VirtualizedSelectFocusedOption {
    background-color: #2F2F3E !important;
    color: #FFFFFF !important;
}
.data-inspector-panel {
    position: fixed;
    top: 0;
    right: -850px;
    width: 800px;
    height: 100vh;
    background-color: #13131A;
    border-left: 1px solid #222230;
    box-shadow: -20px 0px 50px rgba(0,0,0,0.8);
    transition: right 0.4s ease-in-out;
    z-index: 9999;
    padding: 40px;
    box-sizing: border-box;
    overflow-y: auto;
}
.data-inspector-panel.open {
    right: 0;
}
"""
encoded_css = base64.b64encode(dark_css.encode('utf-8')).decode('utf-8')
external_stylesheets = [f"data:text/css;base64,{encoded_css}"]

app = dash.Dash(__name__, title="Premium Quant Dashboard", external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

# Naprawa "białego kontrastu" w komponencie dcc.Dropdown w Tab 5 (Forward Tracker):
# react-select (silnik pod spodem) domyślnie renderuje kontrolkę i menu na jasnym tle,
# niezależnie od stylu wrappera -- trzeba to nadpisać osobnym CSS wstrzykniętym w <head>.
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .dark-dropdown .Select-control, .dark-dropdown .Select-menu-outer, .dark-dropdown .Select-menu,
            .dark-dropdown .VirtualizedSelectOption, .dark-dropdown .Select--single > .Select-control .Select-value,
            .dark-dropdown div[class*="-control"], .dark-dropdown div[class*="-menu"], .dark-dropdown div[class*="-option"] {
                background-color: #161B22 !important;
                color: #FFFFFF !important;
                border-color: #30363D !important;
            }
            .dark-dropdown .Select-placeholder, .dark-dropdown .Select-value-label, .dark-dropdown input,
            .dark-dropdown div[class*="-singleValue"], .dark-dropdown div[class*="-placeholder"] {
                color: #FFFFFF !important;
            }

            /* Ciemny motyw dla dcc.Slider (rc-slider) -- domyślne style renderują tooltip
               i opisy na jasnym tle, niewidoczne na ciemnym tle terminala. */
            .rc-slider-rail { background-color: #262630 !important; }
            .rc-slider-track { background-color: #6F2CFF !important; }
            .rc-slider-handle {
                background-color: #6F2CFF !important;
                border-color: #6F2CFF !important;
                opacity: 1 !important;
            }
            .rc-slider-handle:hover, .rc-slider-handle:focus, .rc-slider-handle-active {
                border-color: #FF8A00 !important;
                box-shadow: 0 0 0 4px rgba(255,138,0,0.2) !important;
            }
            .rc-slider-dot { background-color: #1C1C24 !important; border-color: #30363D !important; }
            .rc-slider-dot-active { border-color: #6F2CFF !important; }
            .rc-slider-mark-text { color: #8A8A98 !important; }
            .rc-slider-mark-text-active { color: #FFFFFF !important; }
            .rc-slider-tooltip { z-index: 9999 !important; }
            .rc-slider-tooltip-inner {
                background-color: #6F2CFF !important;
                color: #FFFFFF !important;
                border-radius: 6px !important;
                box-shadow: none !important;
                font-weight: 600 !important;
            }
            .rc-slider-tooltip-arrow { border-top-color: #6F2CFF !important; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

THEME = {
    "bg_base": "#0B0B0E",       
    "bg_card": "#13131A",       
    "bg_input": "#1C1C24",      
    "border": "#222230",        
    "text_white": "#FFFFFF",    
    "text_dim": "#8A8A98",      
    "purple": "#6F2CFF",        
    "orange": "#FF8A00",        
    "font": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
}

CHART_COLORS = ["#6F2CFF", "#FF8A00", "#00E5FF", "#FF3366", "#00FF66", "#FFD600", "#B200FF", "#FF6600"]
CLUSTER_PALETTE = {1: "#6F2CFF", 2: "#00E5FF", 3: "#FF8A00", 4: "#FF3366", 5: "#00FF66", 6: "#FFD600"}

# ---------------------------------------------------------------------------
# Style dla głównych zakładek (dcc.Tabs / dcc.Tab) -- przekazywane bezpośrednio
# przez oficjalne propsy `style` / `selected_style`, a nie przez CSS klasy.
# Domyślny wygląd dcc.Tab to biały pasek z czarnym tekstem, co na ciemnym
# tle terminala dawało efekt "białe na białym". Ponieważ to propsy komponentu,
# a nie wewnętrzne nazwy klas DOM, to podejście przetrwa też ewentualne
# przyszłe zmiany w bibliotece dcc (w przeciwieństwie do CSS celującego
# w klasy takie jak .tab / .tab--selected, które mogą się zmieniać między
# wersjami Dash -- patrz notatka o dash<4.0 w requirements.txt).
# ---------------------------------------------------------------------------
TABS_CONTAINER_STYLE = {
    "marginBottom": "30px",
    "borderBottom": f"1px solid {THEME['border']}",
}

TAB_STYLE = {
    "backgroundColor": THEME["bg_card"],
    "color": THEME["text_dim"],
    "border": "none",
    "borderBottom": "3px solid transparent",
    "padding": "16px 22px",
    "fontWeight": "600",
    "fontSize": "13px",
    "letterSpacing": "0.02em",
    "fontFamily": THEME["font"],
}

TAB_SELECTED_STYLE = {
    "backgroundColor": THEME["bg_card"],
    "color": THEME["text_white"],
    "border": "none",
    "borderBottom": f"3px solid {THEME['purple']}",
    "padding": "16px 22px",
    "fontWeight": "700",
    "fontSize": "13px",
    "letterSpacing": "0.02em",
    "fontFamily": THEME["font"],
    "boxShadow": f"inset 0 -1px 0 0 {THEME['purple']}",
}

# --- UTILS BEZPIECZEŃSTWA ---
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

# --- SEMI-COVARIANCE ENGINE (Estrada, symetryczna wersja downside risk) ---
def compute_semicovariance_matrix(returns_df):
    """
    SemiCov_ij = (1/T) * sum_t [ min(r_it - mean_i, 0) * min(r_jt - mean_j, 0) ]
    Próg to WŁASNA średnia każdej spółki (nie zero, nie wspólna średnia portfela).
    Górne (dodatnie) odchylenia są zerowane NIEZALEŻNIE dla każdej kolumny,
    dzięki czemu iloczyn macierzowy D^T @ D jest automatycznie symetryczny.
    """
    demeaned = returns_df - returns_df.mean()
    downside = demeaned.clip(upper=0)  # min(x, 0) na całej macierzy naraz
    T = len(downside)
    semicov = (downside.T @ downside) / T
    return semicov

def semicov_to_semicorr(semicov_matrix):
    """Normalizuje semi-kowariancję do przedziału [-1, 1], analogicznie do zwykłej korelacji z kowariancji."""
    diag = np.sqrt(np.diag(semicov_matrix.values))
    denom = np.outer(diag, diag)
    with np.errstate(divide='ignore', invalid='ignore'):
        semicorr = semicov_matrix.values / denom
    semicorr = np.nan_to_num(semicorr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(semicorr, 1.0)
    return pd.DataFrame(semicorr, index=semicov_matrix.index, columns=semicov_matrix.columns)

# --- METODA 1: RMT DENOISED MATRIX (Filtr Marcenko-Pastur) ---
def rmt_denoise_correlation(daily_returns_df):
    X = daily_returns_df.values
    T, N = X.shape
    X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
    C = np.corrcoef(X_std, rowvar=False)

    q = T / N
    lam_plus = (1 + np.sqrt(1 / q)) ** 2  # sigma^2 = 1 (bazowe ujęcie MP dla macierzy korelacji)

    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    is_noise = eigvals <= lam_plus
    eigvals_clean = eigvals.copy()
    if is_noise.sum() > 0:
        eigvals_clean[is_noise] = eigvals[is_noise].mean()

    C_denoised = eigvecs @ np.diag(eigvals_clean) @ eigvecs.T
    d = np.sqrt(np.clip(np.diag(C_denoised), 1e-12, None))
    C_denoised = C_denoised / np.outer(d, d)
    np.fill_diagonal(C_denoised, 1.0)

    return pd.DataFrame(C_denoised, index=daily_returns_df.columns, columns=daily_returns_df.columns), lam_plus, eigvals

# --- METODA 2: DBSCAN — automatyczny dobór eps metodą łokcia (k-distance) ---
def compute_elbow_eps(sorted_distances):
    n = len(sorted_distances)
    if n < 3:
        return float(np.median(sorted_distances)) if n > 0 else 1.0
    x = np.arange(n, dtype=float)
    y = np.asarray(sorted_distances, dtype=float)
    p1, p2 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    line_vec = p2 - p1
    line_len = np.hypot(*line_vec)
    if line_len == 0:
        return float(y[-1])
    dists = np.abs(line_vec[0] * (p1[1] - y) - (p1[0] - x) * line_vec[1]) / line_len
    return float(y[int(np.argmax(dists))])

# --- METODA 3: DTW (Dynamic Time Warping) + K-Medoids ---
def dtw_distance(x, y, window=20):
    """DTW z ograniczeniem pasmowym Sakoe-Chiba: liczymy tylko komórki w promieniu `window`
    wokół przekątnej, zamiast całej macierzy n x m. Redukuje złożoność z O(n*m) do O(n*window)."""
    n, m = len(x), len(y)
    w = max(window, abs(n - m))

    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - w)
        j_end = min(m + 1, i + w + 1)
        xi = x[i - 1]
        for j in range(j_start, j_end):
            cost = (xi - y[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return np.sqrt(D[n, m])

def compute_dtw_distance_matrix(norm_df, window=20):
    tickers = list(norm_df.columns)
    N = len(tickers)
    D = np.zeros((N, N))
    series = [norm_df.iloc[:, i].values for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            d = dtw_distance(series[i], series[j], window=window)
            D[i, j] = D[j, i] = d
    return pd.DataFrame(D, index=tickers, columns=tickers)

def kmedoids(D, k, n_iter=100, random_state=42):
    n = D.shape[0]
    k = min(k, n)
    rng = np.random.RandomState(random_state)
    medoid_idx = list(rng.choice(n, size=k, replace=False))
    for _ in range(n_iter):
        labels = np.argmin(D[:, medoid_idx], axis=1)
        new_medoids = []
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                new_medoids.append(medoid_idx[c])
                continue
            sub = D[np.ix_(members, members)]
            best = members[np.argmin(sub.sum(axis=1))]
            new_medoids.append(int(best))
        if sorted(new_medoids) == sorted(medoid_idx):
            medoid_idx = new_medoids
            break
        medoid_idx = new_medoids
    labels = np.argmin(D[:, medoid_idx], axis=1)
    return labels, medoid_idx

# --- WSPÓLNE BUDOWANIE WIZUALIZACJI (dendrogram + legenda klastrów) ---
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

# ============================================================
# STAGE 3 — PORTFOLIO CONSTRUCTION INPUTS (niezależny moduł, nie dotyka Stage 1/2)
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

STAGE4A_PARAMS_CONFIG = [
    {"id": "lambda", "name": "Lambda", "symbol": "λ", "default": 3.0, "min": 0.5, "max": 15.0, "step": 0.5,
     "comment": "Crash-overlap penalty sensitivity: exp(λ·√(wᵀKw)). Rescaled for the K-based quadratic penalty (Section 3.3/3.4) — √(wᵀKw) typically runs ≈0.05–0.20, so λ needs a much wider range than the old per-asset Z-score exponent did to have a comparable effect."},
    {"id": "gamma", "name": "Gamma", "symbol": "γ", "default": 1.50, "min": 1.00, "max": 2.50,
     "comment": "Analyst range spread penalty. Disincentivizes stocks with wide target price disagreements."},
    {"id": "kappa", "name": "Kappa", "symbol": "κ", "default": 1.00, "min": 0.50, "max": 2.00,
     "comment": "Sensitivity to 90-day EPS consensus revisions (short-term earnings momentum)."},
    {"id": "nref", "name": "N_ref", "symbol": "N_ref", "default": 8.0, "min": 5, "max": 15,
     "comment": "Reference analyst coverage threshold for maximum confidence factor A_i."},
    {"id": "wmax", "name": "Max Asset Weight", "symbol": "w_max", "default": 0.30, "min": 0.10, "max": 0.50, "step": 0.05,
     "comment": "Hard single-stock concentration cap (e.g., 0.30 = max 30% weight per stock). Enforced as a bound directly in the Stage 2 SLSQP; if Stage 1's intra-cluster concentration makes this infeasible, the Singleton Split procedure automatically carves the dominant asset(s) into their own capped micro-cluster — see the alert panel below the results."},
    {"id": "rf", "name": "Risk-Free Rate", "symbol": "R_f", "default": 0.045, "min": 0.0, "max": 0.10,
     "comment": "Annualized risk-free rate (e.g., 10Y Treasury yield) — TPS hurdle rate. Ręczne wejście, brak automatycznego pobierania ^TNX (zgodnie z zasadą 'dane manualne')."},
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

def compute_tail_risk_metrics(raw_close_data):
    """
    Silnik Tail-Risk (Sekcja 4.1):
      DD_i,t = (P_i,t - running_max(P_i)) / running_max(P_i)            [Underwater series]
      TR_i = CDD_0.10,i = -Quantile_0.10({DD_i,t})                       [Empirical quantile drawdown]
      Z_TR,i = (TR_i - mean(TR)) / std(TR)                               [Z-score przekroju uniwersum]
    Penalty P_i = exp(lambda * max(0, Z_TR,i)) liczone OSOBNO (zależy od lambda, patrz compute_tail_penalty).

    DD/CDD liczone przez `ts.compute_drawdown_series` / `ts.compute_cdd_quantile_vec`
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

    dd_df = ts.compute_drawdown_series(prices_df)
    cdd_vec = ts.compute_cdd_quantile_vec(dd_df, quantile=0.10)

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

def compute_composite_upside_row(row, gamma, kappa, n_ref, w1=0.5, w2=0.5):
    """
    Composite Forward Upside (mu_i) — Sekcja 4.2.
    A) Adjusted Target Upside (U_adj):
         U_raw = (T_i - P0) / P0
         S_i   = (T_high - T_low) / T_i                  [uncertainty penalty — rozstrzał widełek]
         A_i   = 1 - exp(-N_i / N_ref)                    [confidence multiplier — pokrycie analityków]
         U_adj = U_raw * exp(-gamma * S_i) * A_i
    B) Adjusted Fundamental EPS Growth Momentum (G_adj):
         G_val   = G_2Y / 100.0        (wejście w tabeli: 55 = 55%)
         Rev_val = Rev_90d / 100.0     (wejście w tabeli: 5 = +5%)
         M_i     = 1 + kappa * Rev_val
         G_adj   = G_val * M_i
    C) mu_i = w1 * U_adj + w2 * G_adj
    Zabezpieczenia przed dzieleniem przez zero: P0<=0 -> U_raw=0; T_i==0 -> S_i=0; N_ref<=0 -> A_i=0.
    """
    P0 = row.get("Current Price (P0)") or 0.0
    Ti = row.get("Target Consensus (Ti)") or 0.0
    Thigh = row.get("Target High (T_high)") or 0.0
    Tlow = row.get("Target Low (T_low)") or 0.0
    Ni = row.get("Analyst Coverage (Ni)") or 0.0
    G2Y_pct = row.get("EPS 2Y CAGR (Gi,2Y)") or 0.0
    Rev_pct = row.get("90d EPS Revision (ΔEPS90d)") or 0.0

    U_raw = (Ti - P0) / P0 if P0 > 0 else 0.0
    S_i = (Thigh - Tlow) / Ti if Ti != 0 else 0.0
    A_i = (1.0 - np.exp(-Ni / n_ref)) if n_ref > 0 else 0.0
    U_adj = U_raw * np.exp(-gamma * S_i) * A_i

    G_val = G2Y_pct / 100.0
    Rev_val = Rev_pct / 100.0
    M_i = 1.0 + kappa * Rev_val
    G_adj = G_val * M_i

    mu_i = w1 * U_adj + w2 * G_adj

    return {"U_raw": float(U_raw), "S_i": float(S_i), "A_i": float(A_i), "U_adj": float(U_adj),
            "G_val": float(G_val), "Rev_val": float(Rev_val), "M_i": float(M_i), "G_adj": float(G_adj),
            "mu_i": float(mu_i)}

# --- LAYOUT APLIKACJI ---
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

                                html.Label("LAMBDA — kara za współkrach (λ, K = J⊙S)", style={"fontSize": "10px", "fontWeight": "bold", "color": THEME["text_dim"], "letterSpacing": "0.5px"}),
                                dcc.Slider(id="slider-sb-lambda", min=0.5, max=15.0, step=0.5, value=3.0, marks={0.5: "0.5", 5: "5", 10: "10", 15: "15"}, tooltip={"placement": "bottom", "always_visible": True}),

                                html.Label("GAMMA — kara za rozstrzał widełek (γ)", style={"fontSize": "10px", "fontWeight": "bold", "color": THEME["text_dim"], "letterSpacing": "0.5px", "marginTop": "20px", "display": "block"}),
                                dcc.Slider(id="slider-sb-gamma", min=0.5, max=3.0, step=0.1, value=1.5, marks={1: "1", 2: "2", 3: "3"}, tooltip={"placement": "bottom", "always_visible": True}),

                                html.Label("KAPPA — momentum rewizji EPS (κ)", style={"fontSize": "10px", "fontWeight": "bold", "color": THEME["text_dim"], "letterSpacing": "0.5px", "marginTop": "20px", "display": "block"}),
                                dcc.Slider(id="slider-sb-kappa", min=0.0, max=3.0, step=0.1, value=1.0, marks={0: "0", 1.5: "1.5", 3: "3"}, tooltip={"placement": "bottom", "always_visible": True}),

                                html.Label("MAX SINGLE WEIGHT (w_max)", style={"fontSize": "10px", "fontWeight": "bold", "color": THEME["text_dim"], "letterSpacing": "0.5px", "marginTop": "20px", "display": "block"}),
                                dcc.Slider(id="slider-sb-wmax", min=0.1, max=0.5, step=0.05, value=0.30, marks={0.1: "10%", 0.3: "30%", 0.5: "50%"}, tooltip={"placement": "bottom", "always_visible": True}),

                                html.Div(style={"borderTop": "1px solid #30363D", "margin": "24px 0 16px 0"}),
                                html.Div("WEIGHT BREAKDOWN: ORIGINAL vs SANDBOX", style={"fontSize": "10px", "fontWeight": "bold", "color": THEME["text_dim"], "letterSpacing": "0.5px", "marginBottom": "10px"}),
                                html.Div(id="sandbox-weights-table-container")
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
                        ])
                    ])
                ])
            ])
        ]),
    ])
])



# --- SYSTEM CALLBACKS ---

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
    Output("dropdown-active-asset", "options"), Output("dropdown-active-asset", "value"), Output("validated-tags-container", "children"),
    Output("panel-preview-container", "style"), Output("panel-matrix-container", "style"), Output("panel-config-stage2-container", "style"),
    Output("matrix-date-range-sub", "children"), Output("table-correlation-wrapper", "children"),
    Output("store-raw-close", "data"), Output("store-monthly-returns", "data"),
    Output("store-daily-returns", "data"), Output("store-semicov-matrix", "data"),
    Output("error-output", "children"),
    Input("btn-validate", "n_clicks"), State("input-tickers-raw", "value")
)
def run_stage_01_ingestion(n_clicks, raw_input):
    if n_clicks == 0 or not raw_input: return dash.no_update, dash.no_update, dash.no_update, {"display": "none"}, {"display": "none"}, {"display": "none"}, "", "", None, None, None, None, ""
    tickers = [t.strip().upper() for t in raw_input.replace(",", " ").split(" ") if t.strip()]

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

        tags = [html.Div(vt, style={"padding": "10px 20px", "background": "rgba(111, 44, 255, 0.12)", "border": f"1px solid {THEME['purple']}", "borderRadius": "24px", "fontSize": "13px"}) for vt in valid_tickers]
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
            columns=columns, data=data_records, style_data_conditional=generate_tws_matrix_styles(grid_df, active_ts),
            style_header={'backgroundColor': '#0B0B0E', 'color': THEME['text_white'], 'fontWeight': 'bold', 'border': '1px solid #222230', 'padding': '12px'},
            style_data={'backgroundColor': THEME['bg_card'], 'color': THEME['text_white'], 'border': '1px solid #1E1E28', 'padding': '12px'},
            style_cell={'textAlign': 'center'},
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
        html.Span(f"Sugerowane k = {best_k} (silhouette score = {scores[best_k]:.3f}). ", style={"color": THEME["purple"], "fontWeight": "bold"}),
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
                dendro_fig.add_trace(go.Scatter(x=list(range(len(nn_dist))), y=nn_dist, mode='lines+markers', line=dict(color=THEME["purple"], width=2), marker=dict(size=5), name="k-distance"))
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
        fig_clustered_map = go.Figure(data=go.Heatmap(
            z=reordered_matrix.values, x=ordered_tickers, y=ordered_tickers, zmin=zmin, zmax=zmax,
            colorscale=[[0.0, THEME["orange"]], [0.5, "#13131A"], [1.0, THEME["purple"]]], showscale=False, hoverongaps=False
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
        style_header={'backgroundColor': '#0B0B0E', 'color': '#FFFFFF', 'fontWeight': 'bold', 'border': '1px solid #222230', 'fontFamily': THEME['font']},
        style_data={'backgroundColor': THEME['bg_input'], 'color': '#FFFFFF', 'border': '1px solid #222230', 'fontFamily': THEME['font']},
        style_cell={'padding': '10px', 'textAlign': 'center'}
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
                     style={"color": THEME["orange"] if missing_price else THEME["purple"], "fontWeight": "bold"})
    return table_data, msg

@app.callback(Output("panel-stage4a-container", "style"), Input("store-stage3-final-payload", "data"), prevent_initial_call=True)
def reveal_stage4a_panel(payload):
    if payload:
        return {"display": "block", "marginBottom": "35px"}
    return {"display": "none", "marginBottom": "35px"}

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

@app.callback(Output("panel-stage4b-container", "style"), Input("store-stage3-final-payload", "data"), prevent_initial_call=True)
def reveal_stage4b_panel(payload):
    if payload:
        return {"display": "block", "marginBottom": "35px", "marginTop": "35px"}
    return {"display": "none", "marginBottom": "35px", "marginTop": "35px"}

@app.callback(Output("panel-stage4b-tracker-container", "style"), Input("store-stage3-final-payload", "data"), prevent_initial_call=True)
def reveal_stage4b_tracker_panel(payload):
    if payload:
        return {"display": "block", "marginBottom": "35px"}
    return {"display": "none", "marginBottom": "35px"}

@app.callback(
    Output("store-stage4b-results", "data"),
    Input("input-lambda", "value"), Input("input-gamma", "value"), Input("input-kappa", "value"), Input("input-nref", "value"),
    Input("input-wmax", "value"), Input("input-rf", "value"),
    Input("store-stage3-final-payload", "data"), Input("store-stage4a-tailrisk", "data"), Input("store-crash-matrices", "data"),
    State("store-daily-returns", "data"), State("store-raw-close", "data"),
    prevent_initial_call=True
)
def run_stage4b_solver(lam, gamma, kappa, n_ref, w_max, rf,
                        stage3_payload, tailrisk, crash_data, daily_returns_data, raw_close_data):
    """
    Silnik Stage 4B — True Two-Stage SLSQP (Section 3.4) + Dynamic Singleton Split.
    Łączy Section 4.2 (mu_i, z Stage 3+4A), macierz Estrady Sigma_eps (Tikhonov-regularyzowaną),
    macierz Discrete Crash-Overlap K (Section 3.3) i ts.run_optimization_with_singleton_split()
    w jeden pipeline dający finalne wagi portfela.

    Stary P_i (Z-score tail penalty) jest TU liczony WYŁĄCZNIE do celów diagnostycznych
    (Standalone Asset TPS w tabeli, wykres underwater w Tab 3) -- solver go nie dotyka,
    zgodnie z decyzją (d)/(e) z ustaleń refaktoryzacji.

    Zwraca dict {"error": "..."} zamiast rzucać wyjątkiem, żeby UI mógł pokazać czytelny komunikat
    zamiast białej strony / cichego zawieszenia callbacku.
    """
    if not stage3_payload or not tailrisk or not tailrisk.get("tickers") or not daily_returns_data or not raw_close_data or not crash_data:
        return {"error": "Brak danych — upewnij się, że Stage 1-3 zostały uruchomione i Stage 3 zatwierdzony (CONFIRM & EXPORT)."}

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
        sigma_full = ts.compute_estrada_matrix(daily_returns_df)

        # Discrete Crash-Overlap Matrix K -- już policzona w store-crash-matrices (Sekcja 3.3), tylko odtwarzamy DataFrame i reindeksujemy do common_tickers
        k_tickers = crash_data["tickers"]
        k_full_all = pd.DataFrame(crash_data["K"], index=k_tickers, columns=k_tickers)
        k_full = k_full_all.loc[common_tickers, common_tickers]

        mu_vec, p_vec = {}, {}
        for t in common_tickers:
            z = tailrisk["z"][t]
            p_vec[t] = compute_tail_penalty(z, lam)  # DIAGNOSTYKA ONLY -- solver tego nie widzi (decyzja d)
            upside = compute_composite_upside_row(stage3_by_ticker[t], gamma, kappa, n_ref)
            mu_vec[t] = upside["mu_i"]
        mu_vec = pd.Series(mu_vec)
        p_vec = pd.Series(p_vec)

        clusters_dict = {}
        for t in common_tickers:
            ck = stage3_by_ticker[t].get("Assigned Cluster", 1)
            clusters_dict.setdefault(ck, []).append(t)

        try:
            split_result = ts.run_optimization_with_singleton_split(mu_vec, sigma_full, k_full, clusters_dict, lam, w_max=w_max, Rf=rf)
        except (ValueError, RuntimeError) as e:
            return {"error": f"SLSQP (True Two-Stage + Singleton Split): {str(e)}"}

        final = split_result["final"]
        weights = final["weights"]

        eval_opt = ts.evaluate_portfolio_performance(weights, daily_returns_df, is_log_returns=True, V0=100.0,
                                                      downside_cov_matrix=sigma_full, P_vec=p_vec, Rf=rf)
        delta_ann_map = ts.semideviation_ann_from_matrix(sigma_full)

        # Standalone Asset TPS -- diagnostyka odłączona od solvera (decyzja e/6): "jak wyglądałaby
        # spółka w izolacji", NIE napędza już alokacji (to był Twój pierwotny "błąd metodologiczny").
        asset_tps = ts.compute_asset_tps(mu_vec, p_vec, delta_ann_map, Rf=rf)

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
                uncapped = ts.run_stage2_inter_cluster_slsqp(mu_vec, sigma_full, k_full, pass_clusters, g_series, lam, w_max=1.0, Rf=rf)
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
        return banner, [], [], empty_fig

    tickers = results["tickers"]
    weights, g_map, tps_map, cluster_of = results["weights"], results["g"], results["asset_tps"], results["cluster_of"]
    promoted_tickers = set(results.get("split", {}).get("promoted_tickers", []))

    banner = html.Div()
    if not results.get("solver_success", True):
        banner = html.Div(f"Solver ostrzeżenie: {results.get('solver_message','')}", style={
            "color": THEME["orange"], "fontWeight": "bold", "padding": "14px", "backgroundColor": THEME["bg_input"],
            "borderRadius": "12px", "marginBottom": "25px", "fontSize": "12px"
        })

    # --- KPI cards ---
    kpi_cards = [
        build_kpi_card("OPTIMIZED EXPECTED RETURN (μ_P)", f"{results['mu_p']*100:+.1f}%"),
        build_kpi_card("ANNUALIZED DOWNSIDE RISK (δ_P)", f"{results['delta_p']*100:.1f}%"),
        build_kpi_card("10% TAIL RISK FLOOR (CDD 0.10,P)", f"-{results['cdd_p']*100:.1f}%", color=THEME["orange"]),
        build_kpi_card("PORTFOLIO TPS SCORE (TPS_P)", f"{results['tps_p']:.2f}", color=THEME["purple"]),
    ]

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

    return banner, kpi_cards, table_rows, donut_fig

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
    Output("graph-stage4b-3d", "figure"),
    Input("store-stage4b-results", "data"), Input("store-stage4a-tailrisk", "data"),
    prevent_initial_call=True
)
def render_stage4b_3d_surface(results, tailrisk):
    """3D risk-reward space: X=delta_ann_i, Y=CDD_0.10_i, Z=mu_i, kolor=klaster, + punkt portfela."""
    fig = go.Figure()
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"], height=650)
    if not results or results.get("error") or not tailrisk:
        return fig

    tickers = results["tickers"]
    mu_map, delta_map = results["mu_i_map"], results["delta_ann_map"]
    cdd_map = tailrisk.get("cdd010", {})
    cluster_of, tps_map = results["cluster_of"], results["asset_tps"]

    valid = [t for t in tickers if t in cdd_map]
    xs = [delta_map[t] for t in valid]
    ys = [cdd_map[t] for t in valid]
    zs = [mu_map[t] for t in valid]
    colors = [CLUSTER_PALETTE.get(((int(cluster_of.get(t, 1)) - 1) % 6) + 1, "#888888") for t in valid]
    hover_text = [
        f"<b>{t}</b><br>Cluster: {cluster_of.get(t, 1)}<br>μ_i: {mu_map[t]*100:+.1f}%<br>"
        f"δ_ann: {delta_map[t]*100:.1f}%<br>CDD 0.10: {cdd_map[t]*100:.1f}%<br>TPS_i: {tps_map.get(t, 0.0):+.2f}"
        for t in valid
    ]

    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs, mode='markers',
        marker=dict(size=7, color=colors, line=dict(color=THEME["bg_card"], width=1)),
        text=hover_text, hoverinfo='text', name="Assets"
    ))

    # "Gold star" portfolio point -- Scatter3d nie ma symbolu star, więc duży złoty diamond z białą obwódką
    # to najbliższy wizualnie odpowiednik w ramach dostępnych symboli marker 3D w Plotly.
    fig.add_trace(go.Scatter3d(
        x=[results["delta_p"]], y=[results["cdd_p"]], z=[results["mu_p"]], mode='markers',
        marker=dict(size=14, color="#FFD700", symbol='diamond', line=dict(color="#FFFFFF", width=2)),
        text=[f"<b>FINAL TPS PORTFOLIO</b><br>μ_P: {results['mu_p']*100:+.1f}%<br>δ_P: {results['delta_p']*100:.1f}%<br>"
              f"CDD 0.10,P: {results['cdd_p']*100:.1f}%<br>TPS_P: {results['tps_p']:.2f}"],
        hoverinfo='text', name="Final Portfolio"
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Annualized Downside Risk (δ_ann)", backgroundcolor=THEME["bg_base"], gridcolor="#1E1E28", color=THEME["text_dim"]),
            yaxis=dict(title="10% Quantile Drawdown (CDD 0.10)", backgroundcolor=THEME["bg_base"], gridcolor="#1E1E28", color=THEME["text_dim"]),
            zaxis=dict(title="Composite Forward Upside (μ_i)", backgroundcolor=THEME["bg_base"], gridcolor="#1E1E28", color=THEME["text_dim"]),
        ),
        margin=dict(l=0, r=0, b=0, t=30), showlegend=True, legend=dict(font=dict(color=THEME["text_white"])), font_family=THEME["font"], height=650
    )
    return fig

@app.callback(
    Output("snapshot-save-status", "children"), Output("store-snapshots-refresh", "data", allow_duplicate=True),
    Input("btn-save-snapshot", "n_clicks"),
    State("input-snapshot-name", "value"), State("store-stage4b-results", "data"),
    State("store-stage3-final-payload", "data"), State("store-stage4a-params", "data"),
    State("store-stage4a-tailrisk", "data"), State("store-snapshots-refresh", "data"),
    prevent_initial_call=True
)
def save_snapshot_callback(n_clicks, snapshot_name, stage4b_results, stage3_payload, stage4a_params, tailrisk, counter):
    if not stage4b_results or stage4b_results.get("error"):
        return html.Span("Brak poprawnych wyników optymalizacji do zapisania.", style={"color": THEME["orange"]}), dash.no_update
    if not stage3_payload:
        return html.Span("Brak danych Stage 3 (fundamentals) do zapisania.", style={"color": THEME["orange"]}), dash.no_update

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
        return html.Span(f"Błąd zapisu: {str(e)}", style={"color": THEME["orange"]}), dash.no_update

    spy_note = "" if "SPY" in entry_prices else " (nie udało się pobrać ceny SPY na benchmark)"
    status = html.Span(f"Zapisano: \"{record['snapshot_name']}\" ({record['snapshot_id']}){spy_note}",
                        style={"color": THEME["purple"], "fontWeight": "bold"})
    return status, (counter or 0) + 1

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

def compute_sandbox_allocation(record, tickers, risk_prices, sb_lambda, sb_gamma, sb_kappa, sb_wmax):
    """
    Przelicza wagi 'sandbox' NA ŻYWO -- dokładnie tym samym silnikiem
    ts.run_optimization_with_singleton_split() (True Two-Stage SLSQP + Section
    3.3 crash-overlap matrix K + Dynamic Singleton Split) co główny solver w
    Tab 4 (run_stage4b_solver) -- "Unifikacja Sandboxa", żadnej osobnej
    uproszczonej ścieżki. P_i (stary Z-score penalty) NIE wchodzi już do
    solvera -- to teraz czysto historyczny artefakt, nieużywany tutaj wcale
    (zastąpiony przez K; zostawiony tylko w Tab 4's diagnostycznej tabeli).

    WAŻNE: `risk_prices` to SUROWE ceny (mogą zawierać NaN -- różne kalendarze
    giełd, patrz `compute_crash_overlap_matrix`), okno 5 lat wstecz (~1260 sesji)
    od dziś, używane WYŁĄCZNIE do estymacji Sigma_eps i K -- celowo NIE to samo
    okno co equity curve (które zaczyna się dopiero w dniu zapisu snapshotu).
    Gdyby ryzyko liczyć tylko z okresu "od zapisu", świeży snapshot (sprzed
    kilku dni) nigdy nie miałby wystarczająco sesji i sandbox zawsze spadałby
    na equal-weight niezależnie od suwaków. Zmiana z 2Y na 5Y (ustalenie z
    refaktoryzacji solvera): K i Sigma_eps mają teraz szansę złapać
    wieloletnie cykle i głębokie historyczne korekty, nie tylko ostatnie 2 lata.

    NaN-y w `risk_prices` (ragged multi-exchange calendars) są celowo NIE
    usuwane przed przekazaniem do `compute_crash_overlap_matrix` -- ta funkcja
    ma własną, poprawną obsługę (Option A, patrz tps_solver.py), a wcześniejsze
    zrzucenie NaN-ów na tym etapie zniszczyłoby dokładnie to, co Option A ma
    chronić. Dla Sigma_eps (macierz Estrady) NADAL potrzebny jest czysty,
    wspólny indeks dat -- stąd `.dropna()` tylko lokalnie, przy budowie
    `returns_window`.

    `cap_weights_iteratively` (poniżej) jest tu teraz WYŁĄCZNIE zapasową siatką
    bezpieczeństwa dla ścieżek fallback (equal-weight przy zbyt małej ilości
    danych) -- na ścieżce solvera w_max jest już wymuszony przez same bounds
    SLSQP + automatyczny Singleton Split, więc w praktyce nic tam nie przycina.

    Zwraca (weights: pd.Series, note: str|None).
    """
    fund_by_ticker = {r["Ticker"]: r for r in record.get("fundamental_inputs", [])}
    cluster_of = record.get("cluster_of", {t: 1 for t in tickers})
    frozen_params = record.get("parameters", {}) or {}
    rf = frozen_params.get("rf", 0.045)
    n_ref = frozen_params.get("nref", 8.0) or 8.0

    mu_vec = pd.Series({t: compute_composite_upside_row(fund_by_ticker.get(t, {}), sb_gamma, sb_kappa, n_ref)["mu_i"] for t in tickers})

    equal_weight_fallback = pd.Series(1.0 / len(tickers), index=tickers)
    note = None

    if risk_prices.shape[1] >= 2 and len(risk_prices) >= 20:
        returns_window = np.log(risk_prices / risk_prices.shift(1)).dropna()
        if len(returns_window) >= 10:
            usable_tickers = list(returns_window.columns)
            clusters_dict = {}
            for t in usable_tickers:
                clusters_dict.setdefault(cluster_of.get(t, 1), []).append(t)
            try:
                sigma_sb = ts.compute_estrada_matrix(returns_window)
                crash_sb = ts.compute_crash_overlap_matrix(risk_prices[usable_tickers], quantile=0.10)
                split_result = ts.run_optimization_with_singleton_split(
                    mu_vec.reindex(usable_tickers), sigma_sb, crash_sb["K"], clusters_dict, sb_lambda, w_max=sb_wmax, Rf=rf
                )
                w_raw = split_result["final"]["weights"].reindex(tickers).fillna(0.0)
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
    return w_final, note

@app.callback(
    Output("kpi-summary-row", "children"), Output("sandbox-weights-table-container", "children"),
    Output("graph-forward-equity-curves", "figure"), Output("graph-asset-returns-bar", "figure"),
    Output("graph-forward-drawdowns", "figure"), Output("snapshot-meta-info", "children"),
    Input("dropdown-snapshot-select", "value"), Input("slider-sb-lambda", "value"), Input("slider-sb-gamma", "value"),
    Input("slider-sb-kappa", "value"),
    Input("slider-sb-wmax", "value"), Input("checklist-benchmarks", "value"), Input("btn-refresh-live", "n_clicks"),
    prevent_initial_call=True
)
def update_forward_tracker(snapshot_id, sb_lambda, sb_gamma, sb_kappa, sb_wmax, benchmarks, _):
    empty_fig = go.Figure()
    empty_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"])
    empty_msg_style = {"color": THEME["text_dim"], "fontSize": "12px"}

    if not snapshot_id:
        return [], html.Div("Wybierz zapisany portfel z listy powyżej.", style=empty_msg_style), empty_fig, empty_fig, empty_fig, ""

    record = snap.get_snapshot(snapshot_id)
    if not record:
        return [], html.Div("Nie znaleziono snapshotu (mógł zostać usunięty).", style={"color": THEME["orange"]}), empty_fig, empty_fig, empty_fig, ""

    created_at_full = record.get("created_at", "")
    created_at = created_at_full[:10] if created_at_full else "?"
    holding_days = snap.holding_days_since(record)
    meta_info = f"Utworzono: {created_at}  •  {holding_days} dni w portfolio"

    orig_weights = pd.Series(record.get("final_weights", {}))
    tickers = list(orig_weights.index)
    if not tickers:
        return [], html.Div("Snapshot nie zawiera żadnych aktywów.", style={"color": THEME["orange"]}), empty_fig, empty_fig, empty_fig, meta_info

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
        return [], html.Div(msg, style={"color": THEME["orange"]}), empty_fig, empty_fig, empty_fig, meta_info

    # RAW (bez dropna!) -- ragged multi-exchange NaN-y są potrzebne compute_crash_overlap_matrix
    # (Option A, per-para przecięcie dat); compute_sandbox_allocation robi lokalny .dropna()
    # tylko tam, gdzie faktycznie potrzebny jest wspólny indeks (Sigma_eps).
    risk_prices_full = full_prices[[t for t in tickers if t in full_prices.columns]].tail(1260)
    prices = full_prices[full_prices.index >= created_at]

    if prices.empty or len(prices) < 2:
        msg = "Za mało sesji giełdowych od dnia zapisu, żeby narysować krzywą equity (za świeży snapshot — wróć za dzień/dwa)."
        return [], html.Div(msg, style={"color": THEME["orange"]}), empty_fig, empty_fig, empty_fig, meta_info

    missing_tickers = [t for t in tickers if t not in prices.columns]
    stock_prices = prices[[t for t in tickers if t in prices.columns]].dropna(how="any")
    if stock_prices.shape[1] == 0 or len(stock_prices) < 2:
        return [], html.Div("Brak wspólnych danych cenowych dla żadnej spółki z portfela.", style={"color": THEME["orange"]}), empty_fig, empty_fig, empty_fig, meta_info

    valid_tickers = list(stock_prices.columns)
    stock_returns = stock_prices.pct_change().fillna(0.0)

    manual_weights, sandbox_note = compute_sandbox_allocation(
        record, valid_tickers, risk_prices_full, sb_lambda, sb_gamma, sb_kappa, sb_wmax
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
    fig_eq.add_trace(go.Scatter(x=orig_eq.index, y=orig_eq.values, mode='lines', name="Original Portfolio (zapisany)", line=dict(color=THEME["purple"], width=3)))
    fig_eq.add_trace(go.Scatter(x=sb_eq.index, y=sb_eq.values, mode='lines', name="Manual Sandbox (suwaki)", line=dict(color=THEME["orange"], width=2.5, dash='dash')))
    if "1N" in (benchmarks or []):
        fig_eq.add_trace(go.Scatter(x=eq_1n_eq.index, y=eq_1n_eq.values, mode='lines', name="Equal Weight (1/N)", line=dict(color="#00E5FF", width=2)))
    if "INV_VOL" in (benchmarks or []):
        fig_eq.add_trace(go.Scatter(x=invvol_eq.index, y=invvol_eq.values, mode='lines', name="Equal Risk (Inv-Vol)", line=dict(color="#00E5A0", width=2)))
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
    colors_bar = ["#00E5A0" if v >= 0 else THEME["orange"] for v in asset_perf.values]
    fig_bar = go.Figure(go.Bar(x=asset_perf.index, y=asset_perf.values, marker_color=colors_bar))
    fig_bar.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"],
        margin=dict(l=30, r=10, t=10, b=30), font_family=THEME["font"],
        yaxis=dict(ticksuffix="%", gridcolor="#1E1E28"), xaxis=dict(showgrid=False, tickfont=dict(size=9))
    )

    orig_dd = (orig_eq - orig_eq.cummax()) / orig_eq.cummax() * 100.0
    sb_dd = (sb_eq - sb_eq.cummax()) / sb_eq.cummax() * 100.0
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=orig_dd.index, y=orig_dd.values, mode='lines', name="Original DD", line=dict(color=THEME["purple"], width=1.5)))
    fig_dd.add_trace(go.Scatter(x=sb_dd.index, y=sb_dd.values, mode='lines', name="Sandbox DD", line=dict(color=THEME["orange"], width=1.5)))
    fig_dd.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_base"],
        margin=dict(l=30, r=10, t=10, b=30), font_family=THEME["font"],
        yaxis=dict(ticksuffix="%", gridcolor="#1E1E28"), xaxis=dict(gridcolor="#1E1E28"), showlegend=False
    )

    tbl_rows = [{
        "Ticker": t, "Original": f"{orig_weights.get(t, 0.0)*100:.1f}%",
        "Sandbox": f"{manual_weights.get(t, 0.0)*100:.1f}%", "Return": f"{asset_perf.get(t, 0.0):+.1f}%"
    } for t in valid_tickers]
    weights_table = dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in ["Ticker", "Original", "Sandbox", "Return"]], data=tbl_rows, page_size=10,
        style_header={'backgroundColor': '#0B0B0E', 'color': THEME['text_white'], 'fontWeight': 'bold', 'border': '1px solid #222230', 'fontSize': '10px'},
        style_data={'backgroundColor': '#161B22', 'color': THEME['text_white'], 'border': '1px solid #30363D', 'fontSize': '11px'},
        style_cell={'padding': '6px', 'textAlign': 'center'}
    )
    weights_children = [weights_table]
    if sandbox_note:
        weights_children.append(html.Div(f"ℹ {sandbox_note}", style={"color": THEME["text_dim"], "fontSize": "10px", "marginTop": "8px"}))
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

    kpi_cards = [
        build_kpi_card("PORTFOLIO RETURN", f"{tot_orig_ret:+.2f}%", sub_str=f"Since {created_at}", color="#00E5A0" if tot_orig_ret >= 0 else THEME["orange"]),
        build_kpi_card("ANNUALIZED VOLATILITY", f"{orig_vol_ann:.2f}%", sub_str="Original portfolio"),
        build_kpi_card("MAX DRAWDOWN", f"{orig_dd.min():.2f}%", sub_str="Peak-to-trough", color=THEME["orange"]),
        build_kpi_card("ALPHA vs 1/N", f"{alpha_vs_1n:+.2f}%", sub_str="Pure weighting edge", color=THEME["purple"] if alpha_vs_1n >= 0 else THEME["orange"]),
        build_kpi_card("ALPHA vs SPY", alpha_vs_spy_str, sub_str="Same holding period"),
    ]

    return kpi_cards, html.Div(weights_children), fig_eq, fig_bar, fig_dd, meta_info

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
    Sekcja 3.3: uruchamia ts.compute_crash_overlap_matrix() (J, S, K) zaraz po
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

    result = ts.compute_crash_overlap_matrix(prices_df, quantile=0.10)
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
            colorscale=[[0.0, "#13131A"], [0.5, THEME["orange"]], [1.0, THEME["purple"]]],
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
        style_header={"backgroundColor": THEME["bg_input"], "color": THEME["text_dim"], "fontWeight": "bold", "border": "none", "fontSize": "11px"},
        style_cell={"backgroundColor": THEME["bg_card"], "color": THEME["text_white"], "border": f"1px solid {THEME['border']}", "padding": "10px", "fontSize": "12px"},
        style_data_conditional=[{'if': {'column_id': 'Crash Risk Contribution'}, 'color': THEME["orange"], 'fontWeight': 'bold'}],
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
    Input("input-gamma", "value"), Input("input-kappa", "value"), Input("input-nref", "value"),
    Input("store-stage4a-tailrisk", "data"), Input("store-stage3-table", "data"),
    prevent_initial_call=True
)
def render_stage4a_summary_table(gamma, kappa, n_ref, tailrisk, stage3_rows):
    lam = LEGACY_DIAGNOSTIC_LAMBDA
    if not tailrisk or not tailrisk.get("tickers"):
        return html.Div("Brak danych. Uruchom Stage 1.", style={"color": THEME["orange"], "padding": "20px"})
    gamma = gamma if isinstance(gamma, (int, float)) else 1.50
    kappa = kappa if isinstance(kappa, (int, float)) else 1.00
    n_ref = n_ref if isinstance(n_ref, (int, float)) and n_ref > 0 else 8.0

    stage3_by_ticker = {r["Ticker"]: r for r in (stage3_rows or [])}

    rows = []
    for t in tailrisk["tickers"]:
        z = tailrisk["z"][t]
        p_i = compute_tail_penalty(z, lam)
        s3_row = stage3_by_ticker.get(t, {})
        upside = compute_composite_upside_row(s3_row, gamma, kappa, n_ref)
        rows.append({
            "Ticker": t, "CDD 0.10": tailrisk["cdd010"][t], "Z-Score": z, "Penalty (Pi)": p_i,
            "U_adj": upside["U_adj"], "G_adj": upside["G_adj"], "mu_i": upside["mu_i"]
        })
    columns = [
        {"name": "Ticker", "id": "Ticker"},
        {"name": "CDD 0.10 (i)", "id": "CDD 0.10", "type": "numeric", "format": {"specifier": ".1%"}},
        {"name": "Z-Score (Z_TR,i)", "id": "Z-Score", "type": "numeric", "format": {"specifier": "+.2f"}},
        {"name": "Penalty (P_i)", "id": "Penalty (Pi)", "type": "numeric", "format": {"specifier": ".2f"}},
        {"name": "U_adj", "id": "U_adj", "type": "numeric", "format": {"specifier": "+.1%"}},
        {"name": "G_adj", "id": "G_adj", "type": "numeric", "format": {"specifier": "+.1%"}},
        {"name": "μ_i (Composite Upside)", "id": "mu_i", "type": "numeric", "format": {"specifier": "+.1%"}},
    ]
    return dash_table.DataTable(
        columns=columns, data=rows, page_size=15, sort_action='native',
        style_header={'backgroundColor': '#0B0B0E', 'color': THEME['text_white'], 'fontWeight': 'bold', 'border': '1px solid #222230', 'padding': '10px', 'fontSize': '12px'},
        style_data={'backgroundColor': THEME['bg_input'], 'color': THEME['text_white'], 'border': '1px solid #222230'},
        style_cell={'padding': '10px', 'textAlign': 'center', 'fontSize': '13px'},
        style_cell_conditional=[{'if': {'column_id': 'Ticker'}, 'fontWeight': 'bold', 'textAlign': 'left', 'color': THEME['purple']}],
        style_data_conditional=[
            {'if': {'filter_query': '{Penalty (Pi)} > 1.5', 'column_id': 'Penalty (Pi)'}, 'color': THEME['orange'], 'fontWeight': 'bold'},
            {'if': {'filter_query': '{mu_i} > 0', 'column_id': 'mu_i'}, 'color': '#00E5A0'},
            {'if': {'filter_query': '{mu_i} < 0', 'column_id': 'mu_i'}, 'color': THEME['orange']}
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
    fig.add_trace(go.Scatter(x=dates, y=dd, mode='lines', fill='tozeroy', line=dict(color=THEME["purple"], width=1.5),
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

if __name__ == "__main__":
    app.run(debug=True, port=8050)