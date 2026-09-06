"""
engine/clustering.py
=====================
Pure math for asset clustering: Estrada semi-covariance (legacy demeaned
variant, distinct from the regularized downside-only Sigma_eps in
engine/risk.py), Marchenko-Pastur RMT denoising, DBSCAN elbow-epsilon
detection, and DTW distance + K-Medoids partitioning.

No Dash / plotting code lives here on purpose -- figure-building helpers
(build_dendrogram_figure, build_cluster_legend) stay in ui/, since they
return Plotly figures / Dash components, not numbers.

Moved out of quant_terminal.py (Etap 0 architecture split,
PROJECT_CONTEXT.md) with NO behavior change -- verbatim relocation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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
