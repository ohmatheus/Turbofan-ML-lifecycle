import numpy as np
import pandas as pd
from scipy import stats  # type: ignore[import-untyped]


def compute_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    baseline = np.asarray(baseline).astype(float)
    current = np.asarray(current).astype(float)

    # Define bins based on baseline quantiles
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(baseline, quantiles)

    # Ensure strictly increasing bin edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) - 1 < n_bins:
        # fallback: uniform bins between min and max
        bin_edges = np.linspace(baseline.min(), baseline.max(), n_bins + 1)

    baseline_hist, _ = np.histogram(baseline, bins=bin_edges)
    current_hist, _ = np.histogram(current, bins=bin_edges)

    baseline_prop = baseline_hist / max(baseline_hist.sum(), epsilon)
    current_prop = current_hist / max(current_hist.sum(), epsilon)

    psi_values = (baseline_prop - current_prop) * np.log((baseline_prop + epsilon) / (current_prop + epsilon))
    return float(np.sum(psi_values))


def compute_ks(baseline: np.ndarray, current: np.ndarray) -> float:
    baseline = np.asarray(baseline).astype(float)
    current = np.asarray(current).astype(float)
    statistic, _ = stats.ks_2samp(baseline, current)
    return float(statistic)


def compute_feature_psi(
    df_baseline: pd.DataFrame,
    df_current: pd.DataFrame,
    feature_cols: list[str],
    n_bins: int = 10,
) -> dict[str, float]:
    psi_scores: dict[str, float] = {}
    for col in feature_cols:
        psi_scores[col] = compute_psi(df_baseline[col].to_numpy(), df_current[col].to_numpy(), n_bins=n_bins)
    return psi_scores


def compute_feature_ks(
    df_baseline: pd.DataFrame,
    df_current: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, float]:
    ks_scores: dict[str, float] = {}
    for col in feature_cols:
        ks_scores[col] = compute_ks(df_baseline[col].to_numpy(), df_current[col].to_numpy())
    return ks_scores


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
