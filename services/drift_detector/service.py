import json
import time
import threading
from pathlib import Path
from typing import Any
from datetime import datetime, timedelta, timezone

import bentoml
import numpy as np
from prometheus_client import Counter, Gauge
from pydantic import BaseModel

from src.models.random_forest_utils import rmse_score
from src.utils.config import config

# How often to run drift checks (seconds)
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
        #todo baseline rmse will be available via the retraining service



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
            time.sleep(DRIFT_CHECK_INTERVAL_SECONDS)

    def _load_recent_feedback(self, window_seconds: int = 60) -> list[dict[str, Any]]:
        """
        Load feedback records whose feedback_timestamp is within the last `window_seconds`.
        Default is last 60 seconds.
        """
        if not self.feedback_storage_path.exists():
            return []

        now = datetime.now(timezone.utc)
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
                        ts = ts.replace(tzinfo=timezone.utc)
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

        current_rmse = rmse_score(y_true, y_pred)

        # Initialize baseline if not set yet
        if self._baseline_rmse is None:
            self._baseline_rmse = float(current_rmse)
            drift_rmse_baseline.set(self._baseline_rmse)

        # Simple alert heuristic: 30% worse than baseline
        alert_threshold_factor = 1.3
        is_alert = bool(current_rmse > self._baseline_rmse * alert_threshold_factor)

        # Update Prometheus metrics
        drift_checks_total.inc()
        drift_rmse_current.set(current_rmse)
        drift_rmse_baseline.set(self._baseline_rmse)
        drift_rmse_alert.set(1.0 if is_alert else 0.0)

        # Update in-memory status
        self._last_status = DriftStatus(
            last_check_time=datetime.now(timezone.utc).isoformat(),
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