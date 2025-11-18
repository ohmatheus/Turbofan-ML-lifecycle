import json
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import bentoml
import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge
from pydantic import BaseModel

from src.drift.logic import evaluate_drift
from src.drift.metrics import (
    compute_feature_ks,
    compute_feature_psi,
)
from src.drift.thresholds import DEFAULT_THRESHOLDS
from src.models.random_forest_utils import EXCLUDE_COLS, rmse_score
from src.utils.config import config

TEMP_TRAIN_RATIO = 0.1  # todo replace with exposed by retraining service

DRIFT_CHECK_INTERVAL_SECONDS = 15

drift_checks_total = Counter(
    name="rul_drift_checks_total",
    documentation="Total number of drift checks performed",
)
drift_rmse_current = Gauge(
    name="rul_drift_rmse_current",
    documentation="Current RMSE computed from recent feedback",
)
drift_rmse_baseline = Gauge(
    name="rul_drift_rmse_baseline",
    documentation="Baseline RMSE used for drift comparison",
)
drift_rmse_alert_threshold = Gauge(
    name="rul_drift_rmse_alert_threshold",
    documentation="Alert RMSE used for retraining",
)
drift_rmse_alert = Gauge(
    name="rul_drift_rmse_alert",
    documentation="1 if current RMSE is above alert threshold, 0 otherwise",
)


class DriftStatus(BaseModel):
    last_check_time: str | None = None
    current_rmse: float | None = None
    baseline_rmse: float | None = None
    alert: bool = False


@bentoml.service(
    resources={"cpu": "1", "memory": "1Gi"},
    traffic={"timeout": 5, "concurrency": 50},
)
class DriftDetectorService:
    def __init__(self) -> None:
        self.feedback_storage_path = Path(config.FEEDBACK_PATH / "rul_feedback.jsonl")
        self.feedback_storage_path.parent.mkdir(parents=True, exist_ok=True)

        self._baseline_rmse: float | None = getattr(config, "BASELINE_RMSE", None)
        # todo baseline rmse will be available via the retraining service

        # not optimal in production to save data source here, but simpler for demo - would need otherwise make a query to data source and a api that manage this
        self.train_df = pd.read_csv(config.READY_DATA_PATH / "train.csv", index_col=False)
        self.test_df = pd.read_csv(config.READY_DATA_PATH / "test_last_rows.csv", index_col=False)

        self._last_status = DriftStatus()

        # Set initial Prometheus values if baseline exists
        if self._baseline_rmse is not None:
            drift_rmse_baseline.set(self._baseline_rmse)

        self._start_background_loop()

    def _start_background_loop(self) -> None:
        thread = threading.Thread(target=self._drift_loop, daemon=True)
        thread.start()

    def _drift_loop(self) -> None:
        """Internal loop that runs drift checks at a fixed interval."""
        while True:
            try:
                self._run_drift_check()
            except Exception as exc:  # noqa: BLE001
                # Log and continue; we don't want to kill the loop
                print(f"[DriftDetector] Error during drift check: {exc}")
                drift_checks_total.inc()
                drift_rmse_current.set(0)
                drift_rmse_baseline.set(0)
                drift_rmse_alert_threshold.set(0)
                drift_rmse_alert.set(0)
            time.sleep(DRIFT_CHECK_INTERVAL_SECONDS)

    def _load_recent_feedback(self, window_seconds: int = 60) -> list[dict[str, Any]]:
        """
        Load feedback records whose feedback_timestamp is within the last `window_seconds`.
        Default is last 60 seconds.
        """
        if not self.feedback_storage_path.exists():
            return []

        now = datetime.now(UTC)
        window_start = now - timedelta(seconds=window_seconds)

        records: list[dict[str, Any]] = []
        with open(self.feedback_storage_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip malformed lines

                ts_str = record.get("feedback_timestamp")
                if not ts_str:
                    continue

                try:
                    # Parse as UTC-aware datetime
                    # handle both "...Z" and explicit offset forms
                    if ts_str.endswith("Z"):
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    else:
                        ts = datetime.fromisoformat(ts_str)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=UTC)
                except ValueError:
                    continue

                if ts >= window_start:
                    records.append(record)

        return records

    def _run_drift_check(self) -> None:
        """Compute RMSE from recent feedback and update metrics/status."""
        feedback_records = self._load_recent_feedback(window_seconds=60)
        if not feedback_records:
            print("[DriftDetector] No feedback records found, skipping drift check")
            return

        y_true = np.asarray([fb["actual_rul"] for fb in feedback_records], dtype=float)
        y_pred = np.asarray([fb["predicted_rul"] for fb in feedback_records], dtype=float)
        current_rmse = rmse_score(y_true, y_pred)  # accumulated of last minute predictions

        # Initialize baseline if not set yet - todo redo
        if self._baseline_rmse is None:
            self._baseline_rmse = float(current_rmse)
            drift_rmse_baseline.set(self._baseline_rmse)

        rmse_alert_threshold = self._baseline_rmse * DEFAULT_THRESHOLDS.rmse_alert_factor
        is_alert = bool(current_rmse > rmse_alert_threshold)

        # Update Prometheus metrics
        drift_checks_total.inc()
        drift_rmse_current.set(current_rmse)
        drift_rmse_baseline.set(self._baseline_rmse)
        drift_rmse_alert_threshold.set(rmse_alert_threshold)
        drift_rmse_alert.set(1.0 if is_alert else 0.0)

        # psi - ks
        unit_ids = {fb["engine_id"] for fb in feedback_records}  # all unit_number of last minute predictions
        max_unit_id = max(unit_ids)
        recent_units_train_df = self.train_df[self.train_df["unit_number"].isin(unit_ids)]
        recent_units_test_df = self.test_df[self.test_df["unit_number"].isin(unit_ids)]

        train_mask = self.train_df["unit_number"] <= max_unit_id
        train_up_to_max_unit_df = self.train_df.loc[train_mask].copy()

        test_mask = self.test_df["unit_number"] <= max_unit_id
        test_up_to_max_unit_df = self.test_df.loc[test_mask].copy()

        temp_size = int(len(self.train_df) * TEMP_TRAIN_RATIO)
        train_df_used_for_training = self.train_df.iloc[:temp_size]

        feature_cols = [col for col in self.train_df.columns if col not in EXCLUDE_COLS]

        psi_scores_train = compute_feature_psi(train_df_used_for_training, train_up_to_max_unit_df, feature_cols)
        ks_scores_train = compute_feature_ks(train_df_used_for_training, train_up_to_max_unit_df, feature_cols)

        psi_scores_test = compute_feature_psi(train_df_used_for_training, recent_units_test_df, feature_cols)
        ks_scores_test = compute_feature_ks(train_df_used_for_training, recent_units_test_df, feature_cols)

        drift_report_train = evaluate_drift(
            psi_per_feature=psi_scores_train,
            ks_per_feature=ks_scores_train,
            current_rmse=current_rmse,
            baseline_rmse=self._baseline_rmse,
            thresholds=DEFAULT_THRESHOLDS,
        )

        drift_report_test = evaluate_drift(
            psi_per_feature=psi_scores_train,
            ks_per_feature=ks_scores_train,
            current_rmse=current_rmse,
            baseline_rmse=self._baseline_rmse,
            thresholds=DEFAULT_THRESHOLDS,
        )

        # Update in-memory status
        self._last_status = DriftStatus(
            last_check_time=datetime.now(UTC).isoformat(),
            current_rmse=float(current_rmse),
            baseline_rmse=float(self._baseline_rmse),
            alert=is_alert,
        )

        print(
            f"[DriftDetector] Drift check at {self._last_status.last_check_time}: "
            f"current_rmse={current_rmse:.4f}, "
            f"baseline_rmse={self._baseline_rmse:.4f}, "
            f"alert={is_alert}",
        )

        # TODO: when is_alert is True, call retraining service /retrain endpoint.

    @bentoml.api
    def run_drift_check_now(self) -> dict[str, Any]:
        """
        Manually trigger a drift check and return the latest status.
        For debugging or on-demand checks.
        """
        self._run_drift_check()
        return self._last_status.model_dump()

    @bentoml.api
    def get_drift_status(self) -> dict[str, Any]:
        return self._last_status.model_dump()
