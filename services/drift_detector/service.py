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
from src.drift.metrics import compute_feature_ks, compute_feature_psi
from src.drift.thresholds import DEFAULT_THRESHOLDS
from src.models.random_forest_utils import EXCLUDE_COLS, rmse_score
from src.utils.config import config

TEMP_TRAIN_RATIO = 0.1
DRIFT_CHECK_INTERVAL_SECONDS = 15


drift_checks_total = Counter("rul_drift_checks_total", "Total number of drift checks performed")
drift_rmse_current = Gauge("rul_drift_rmse_current", "Current RMSE computed from recent feedback")
drift_rmse_baseline = Gauge("rul_drift_rmse_baseline", "Baseline RMSE used for drift comparison")
drift_rmse_warn_threshold = Gauge("rul_drift_rmse_warn_threshold", "Alert RMSE used for retraining")
drift_rmse_alert_threshold = Gauge("rul_drift_rmse_alert_threshold", "Alert RMSE used for retraining")
drift_rmse_alert = Gauge("rul_drift_rmse_alert", "1 if current RMSE is above alert threshold, 0 otherwise")

psi_mean_gauge = Gauge(
    "rul_drift_psi_mean",
    "Mean PSI across all features",
    labelnames=("dataset",), #todo remove
)
drift_psi_warn_threshold = Gauge("rul_drift_psi_warn_threshold", "Warning PSI used for retraining")
drift_psi_alert_threshold = Gauge("rul_drift_psi_alert_threshold", "Alert PSI used for retraining")
drift_psi_alert = Gauge("rul_drift_psi_alert", "1 if current PSI is above alert threshold, 0 otherwise")

ks_mean_gauge = Gauge(
    "rul_drift_ks_mean",
    "Mean KS statistic across all features",
    labelnames=("dataset",), #todo remove
)
drift_ks_warn_threshold = Gauge("rul_drift_ks_warn_threshold", "Warning PSI used for retraining")
drift_ks_alert_threshold = Gauge("rul_drift_ks_alert_threshold", "Alert PSI used for retraining")
drift_ks_alert = Gauge("rul_drift_ks_alert", "1 if current PSI is above alert threshold, 0 otherwise")


class DriftStatus(BaseModel):
    last_check_time: str | None = None
    current_rmse: float | None = None
    baseline_rmse: float | None = None
    alert: bool = False


def _now_utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def _feedback_window(path: Path, seconds: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    window_start = datetime.now(UTC) - timedelta(seconds=seconds)
    out: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            if not (line := line.strip()):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not (ts := rec.get("feedback_timestamp")):
                continue
            try:
                ts_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if ts_dt.tzinfo is None:
                    ts_dt = ts_dt.replace(tzinfo=UTC)
            except ValueError:
                continue
            if ts_dt >= window_start:
                out.append(rec)
    return out


def _compute_rmse(feedback: list[dict[str, Any]]) -> float:
    y_true = np.asarray([fb["actual_rul"] for fb in feedback], dtype=float)
    y_pred = np.asarray([fb["predicted_rul"] for fb in feedback], dtype=float)
    return float(rmse_score(y_true, y_pred))


def _training_subset(train_df: pd.DataFrame) -> pd.DataFrame:
    size = int(len(train_df) * TEMP_TRAIN_RATIO)
    return train_df.iloc[:size]


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def _recent_units_df(df: pd.DataFrame, unit_ids: set[int]) -> pd.DataFrame:
    return df[df["unit_number"].isin(unit_ids)]


def _upto_max_unit_df(df: pd.DataFrame, max_unit: int) -> pd.DataFrame:
    return df.loc[df["unit_number"] <= max_unit].copy()


def _compute_psi_ks(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    psi = compute_feature_psi(baseline_df, current_df, features)
    ks = compute_feature_ks(baseline_df, current_df, features)
    return psi, ks


def _set_rmse_metrics(current: float, baseline: float, warn_thresh: float, alert_thresh: float, alert: bool) -> None:
    drift_checks_total.inc()
    drift_rmse_current.set(current)
    drift_rmse_baseline.set(baseline)
    drift_rmse_alert_threshold.set(alert_thresh)
    drift_rmse_warn_threshold.set(warn_thresh)
    drift_rmse_alert.set(1.0 if alert else 0.0)


def _set_mean_drift_metrics(dataset: str, drift_psi_warn_thresh: float, drift_psi_alert_thresh: float, drift_ks_warn_thresh: float, drift_ks_alert_thresh: float, avg_psi: float, avg_ks: float) -> None:
    drift_psi_warn_threshold.set(drift_psi_warn_thresh)
    drift_psi_alert_threshold.set(drift_psi_alert_thresh)
    drift_ks_warn_threshold.set(drift_ks_warn_thresh)
    drift_ks_alert_threshold.set(drift_ks_alert_thresh)
    psi_mean_gauge.labels(dataset=dataset).set(avg_psi)
    ks_mean_gauge.labels(dataset=dataset).set(avg_ks)
    psi_alert = avg_psi > drift_psi_alert_thresh
    ks_alert = avg_ks > drift_ks_alert_thresh
    drift_psi_alert.set(1.0 if psi_alert else 0.0)
    drift_ks_alert.set(1.0 if ks_alert else 0.0)

def reset_gauges():
    drift_checks_total.inc()
    drift_rmse_current.set(0)
    drift_rmse_baseline.set(0)
    drift_rmse_alert_threshold.set(0)
    drift_rmse_alert.set(0)
    psi_mean_gauge.labels(dataset='train').set(0)
    drift_psi_warn_threshold.set(0)
    drift_psi_alert_threshold.set(0)
    drift_psi_alert.set(0)
    ks_mean_gauge.labels(dataset='train').set(0)
    drift_ks_warn_threshold.set(0)
    drift_ks_alert_threshold.set(0)
    drift_ks_alert.set(0)

@bentoml.service(resources={"cpu": "1", "memory": "1Gi"}, traffic={"timeout": 5, "concurrency": 50})
class DriftDetectorService:
    def __init__(self) -> None:
        self.feedback_storage_path = Path(config.FEEDBACK_PATH / "rul_feedback.jsonl")
        self.feedback_storage_path.parent.mkdir(parents=True, exist_ok=True)

        self._baseline_rmse: float | None = getattr(config, "BASELINE_RMSE", None)
        self.train_df = pd.read_csv(config.READY_DATA_PATH / "train.csv", index_col=False)
        self.test_df = pd.read_csv(config.READY_DATA_PATH / "test_last_rows.csv", index_col=False)
        self._last_status = DriftStatus()

        if self._baseline_rmse is not None:
            drift_rmse_baseline.set(self._baseline_rmse)

        self._start_background_loop()

    def _start_background_loop(self) -> None:
        threading.Thread(target=self._drift_loop, daemon=True).start()

    def _drift_loop(self) -> None:
        while True:
            try:
                self._run_drift_check()
            except Exception as exc:  # noqa: BLE001
                print(f"[DriftDetector] Error during drift check: {exc}")
            time.sleep(DRIFT_CHECK_INTERVAL_SECONDS)

    def _run_drift_check(self) -> None:
        feedback = _feedback_window(self.feedback_storage_path, seconds=60)
        if not feedback:
            print("[DriftDetector] No feedback records found, skipping drift check")
            reset_gauges()
            return

        current_rmse = _compute_rmse(feedback)
        if self._baseline_rmse is None:
            self._baseline_rmse = float(current_rmse)
            drift_rmse_baseline.set(self._baseline_rmse)

        alert_thr = self._baseline_rmse * DEFAULT_THRESHOLDS.rmse_alert_factor
        warn_thr = self._baseline_rmse * DEFAULT_THRESHOLDS.rmse_warn_factor
        is_alert = current_rmse > warn_thr
        _set_rmse_metrics(current_rmse, self._baseline_rmse, warn_thr, alert_thr, bool(is_alert))

        unit_ids = {fb["engine_id"] for fb in feedback}
        max_unit = max(unit_ids)

        train_used = _training_subset(self.train_df)
        feats = _feature_columns(self.train_df)

        train_up_to = _upto_max_unit_df(self.train_df, max_unit)
        test_up_to = _upto_max_unit_df(self.test_df, max_unit)

        train_recent = _recent_units_df(self.train_df, unit_ids)
        test_recent = _recent_units_df(self.test_df, unit_ids)

        #psi_train, ks_train = _compute_psi_ks(train_used, train_up_to, feats)
        #psi_train, ks_train = _compute_psi_ks(train_used, train_recent, feats)
        #psi_test, ks_test = _compute_psi_ks(train_used, test_up_to, feats)
        psi_test, ks_test = _compute_psi_ks(train_used, test_recent, feats)

        # drift_report_train = evaluate_drift(
        #     psi_per_feature=psi_train,
        #     ks_per_feature=ks_train,
        #     current_rmse=current_rmse,
        #     baseline_rmse=self._baseline_rmse,
        #     thresholds=DEFAULT_THRESHOLDS,
        # )

        drift_report_test = evaluate_drift(
            psi_per_feature=psi_test,
            ks_per_feature=ks_test,
            current_rmse=current_rmse,
            baseline_rmse=self._baseline_rmse,
            thresholds=DEFAULT_THRESHOLDS,
        )

        # _set_mean_drift_metrics(
        #     "train",
        #     DEFAULT_THRESHOLDS.psi_warn,
        #     DEFAULT_THRESHOLDS.psi_alert,
        #     DEFAULT_THRESHOLDS.ks_warn,
        #     DEFAULT_THRESHOLDS.ks_alert,
        #     float(drift_report_train["psi"]["avg_psi"]),
        #     float(drift_report_train["ks"]["avg_ks"]),
        # )
        _set_mean_drift_metrics(
            "test",
            DEFAULT_THRESHOLDS.psi_warn,
            DEFAULT_THRESHOLDS.psi_alert,
            DEFAULT_THRESHOLDS.ks_warn,
            DEFAULT_THRESHOLDS.ks_alert,
            float(drift_report_test["psi"]["avg_psi"]),
            float(drift_report_test["ks"]["avg_ks"]),
        )

        self._last_status = DriftStatus(
            last_check_time=_now_utc_iso(),
            current_rmse=float(current_rmse),
            baseline_rmse=float(self._baseline_rmse),
            alert=bool(is_alert),
        )

        print(
            f"[DriftDetector] Drift check at {self._last_status.last_check_time}: "
            f"current_rmse={current_rmse:.4f}, baseline_rmse={self._baseline_rmse:.4f}, alert={bool(is_alert)}"
        )

    @bentoml.api
    def run_drift_check_now(self) -> dict[str, Any]:
        self._run_drift_check()
        return self._last_status.model_dump()

    @bentoml.api
    def get_drift_status(self) -> dict[str, Any]:
        return self._last_status.model_dump()
