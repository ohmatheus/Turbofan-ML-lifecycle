from typing import Any

import pandas as pd

from src.drift.logic import evaluate_drift
from src.drift.metrics import (
    compute_feature_ks,
    compute_feature_psi,
)
from src.drift.thresholds import DEFAULT_THRESHOLDS
from src.models.random_forest_utils import EXCLUDE_COLS
from src.utils.config import config


def log_drift_decision(decision: dict[str, Any]) -> None:
    print("=== Drift Decision ===")
    print(f"Should retrain: {decision['should_retrain']}")

    # PSI summary
    psi = decision["psi"]
    print("\n[PSI]")
    print(f"  avg_psi         : {psi['avg_psi']:.4f}")
    print(f"  max_psi         : {psi['max_psi']:.4f}")
    print(f"  psi_alert_ratio : {psi['psi_alert_ratio']:.4f}")
    print(f"  n_warn_features : {len(psi['warn_features'])}")
    print(f"  n_alert_features: {len(psi['alert_features'])}")

    # Optionally print top-K features by PSI
    scores_psi = psi["scores"]
    if scores_psi:
        top_k = 5
        top_psi = sorted(scores_psi.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        print(f"  Top {top_k} PSI features:")
        for feat, val in top_psi:
            print(f"    - {feat}: {val:.4f}")

    # KS summary
    ks = decision["ks"]
    print("\n[KS]")
    print(f"  avg_ks          : {ks['avg_ks']:.4f}")
    print(f"  max_ks          : {ks['max_ks']:.4f}")
    print(f"  ks_alert_ratio  : {ks['ks_alert_ratio']:.4f}")
    print(f"  n_warn_features : {len(ks['warn_features'])}")
    print(f"  n_alert_features: {len(ks['alert_features'])}")

    scores_ks = ks["scores"]
    if scores_ks:
        top_k = 5
        top_ks = sorted(scores_ks.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        print(f"  Top {top_k} KS features:")
        for feat, val in top_ks:
            print(f"    - {feat}: {val:.4f}")

    # RMSE summary
    rmse = decision["rmse"]
    print("\n[RMSE]")
    print(f"  current : {rmse['current']:.4f}")
    print(f"  baseline: {rmse['baseline']:.4f}")
    print(f"  warn    : {rmse['warn']}")
    print(f"  alert   : {rmse['alert']}")
    print("======================\n")


def main() -> None:
    train_df = pd.read_csv(config.READY_DATA_PATH / "train.csv", index_col=False)
    test_df = pd.read_csv(config.READY_DATA_PATH / "test.csv", index_col=False)

    fd001_train = train_df[train_df["subset"] == 1]
    # fd002_train = train_df[train_df["subset"] == 2]

    fd001_test = test_df[test_df["subset"] == 1]
    # fd002_test = test_df[test_df["subset"] == 2]

    feature_cols = [col for col in train_df.columns if col not in EXCLUDE_COLS]

    psi_scores = compute_feature_psi(fd001_train, fd001_test, feature_cols)
    ks_scores = compute_feature_ks(fd001_train, fd001_test, feature_cols)

    # print("PSI scores:")
    # for f, v in psi_scores.items():
    #     print(f"{f}: {v:.4f}")
    #
    # print("KS scores:")
    # for f, v in ks_scores.items():
    #     print(f"{f}: {v:.4f}")

    drift_report = evaluate_drift(
        psi_per_feature=psi_scores,
        ks_per_feature=ks_scores,
        current_rmse=0,
        baseline_rmse=0,
        thresholds=DEFAULT_THRESHOLDS,
    )

    log_drift_decision(drift_report)


if __name__ == "__main__":
    main()
