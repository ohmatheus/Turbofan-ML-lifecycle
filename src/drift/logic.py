from typing import Any

import numpy as np
import pandas as pd

from src.drift.thresholds import DriftThresholds


# slice_baseline_and_current with train_df, trained_unit_ids, new_units.
# new_units = recent_test_units - trained_unit_ids.
# OR new_units = recent_test_units - trained_unit_ids.
def slice_baseline_and_current(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    trained_unit_ids: list[int],
    new_units: list[int],
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_df = train_df[train_df["unit_id"].isin(trained_unit_ids)][feature_cols]
    current_df = train_df[train_df["unit_id"].isin(new_units)][feature_cols]
    # OR
    # current_df = test_df[test_df["unit_id"].isin(new_units)][feature_cols]

    return baseline_df, current_df


def evaluate_drift(
    psi_per_feature: dict[str, float],
    ks_per_feature: dict[str, float],
    current_rmse: float,
    baseline_rmse: float,
    thresholds: DriftThresholds,
) -> dict[str, Any]:
    """
    Decide whether to retrain based on:
      - distribution drift (PSI, KS) across many features
      - performance drift (RMSE vs baseline)
    """
    # Per-feature alert/warn lists (useful for diagnostics)
    psi_alert_features = [f for f, v in psi_per_feature.items() if v >= thresholds.psi_alert]
    psi_warn_features = [f for f, v in psi_per_feature.items() if thresholds.psi_warn <= v < thresholds.psi_alert]

    ks_alert_features = [f for f, v in ks_per_feature.items() if v >= thresholds.ks_alert]
    ks_warn_features = [f for f, v in ks_per_feature.items() if thresholds.ks_warn <= v < thresholds.ks_alert]

    # Global / average drift metrics
    n_features = len(psi_per_feature) if psi_per_feature else 0
    psi_values = list(psi_per_feature.values())
    ks_values = list(ks_per_feature.values())

    avg_psi = float(np.mean(psi_values)) if psi_values else 0.0
    max_psi = float(np.max(psi_values)) if psi_values else 0.0
    avg_ks = float(np.mean(ks_values)) if ks_values else 0.0
    max_ks = float(np.max(ks_values)) if ks_values else 0.0

    psi_alert_ratio = len(psi_alert_features) / n_features if n_features > 0 else 0.0
    ks_alert_ratio = len(ks_alert_features) / n_features if n_features > 0 else 0.0

    # Heuristics for "global" drift
    psi_alert_ratio_threshold = 0.05
    avg_alert_ratio_threshold = 0.05
    max_psi_alert_threshold = thresholds.psi_alert  # if any feature is really high

    ks_alert_ratio_threshold = 0.05
    avg_ks_alert_threshold = thresholds.ks_warn  # average KS is at least "warn"
    max_ks_alert_threshold = thresholds.ks_alert  # any feature in KS alert

    # has_psi_alert = (
    #     psi_alert_ratio >= psi_alert_ratio_threshold and avg_psi >= avg_alert_ratio_threshold
    # ) or max_psi >= max_psi_alert_threshold
    has_psi_alert = avg_psi >= thresholds.psi_warn
    # has_ks_alert = (
    #     ks_alert_ratio >= ks_alert_ratio_threshold and avg_ks >= avg_ks_alert_threshold
    # ) or max_ks >= max_ks_alert_threshold
    has_ks_alert = avg_ks >= thresholds.ks_warn

    # --- RMSE drift ---
    rmse_warn = current_rmse > baseline_rmse * thresholds.rmse_warn_factor
    rmse_alert = current_rmse > baseline_rmse * thresholds.rmse_alert_factor

    should_retrain = bool(has_psi_alert or has_ks_alert or rmse_warn)

    return {
        "should_retrain": should_retrain,
        "psi": {
            "scores": psi_per_feature,
            "warn_features": psi_warn_features,
            "alert_features": psi_alert_features,
            "avg_psi": avg_psi,
            "max_psi": max_psi,
            "psi_alert_ratio": psi_alert_ratio,
        },
        "ks": {
            "scores": ks_per_feature,
            "warn_features": ks_warn_features,
            "alert_features": ks_alert_features,
            "avg_ks": avg_ks,
            "max_ks": max_ks,
            "ks_alert_ratio": ks_alert_ratio,
        },
        "rmse": {
            "current": current_rmse,
            "baseline": baseline_rmse,
            "warn": rmse_warn,
            "alert": rmse_alert,
        },
    }
