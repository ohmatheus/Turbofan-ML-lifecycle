import gc
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import bentoml
import mlflow
import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge
from sklearn.pipeline import Pipeline

from src.models.model_bundle import ModelMetadata, load_model_bundle, save_model_bundle
from src.models.random_forest_utils import EXCLUDE_COLS, Metrics, eval_rul, fit_rf, plot_rmse
from src.utils.config import config

RETRAIN_COOLDOWN_SECONDS = 30
_last_retrain_ts: float | None = None

LOCK_FILE = Path("/tmp/retrain.lock")


@contextmanager
def try_acquire_train_lock() -> Iterator[bool]:
    fd = None
    try:
        # O_EXCL + O_CREAT -> fail if file already exists
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(time.time()).encode("utf-8"))
        yield True  # lock acquired
    except FileExistsError:
        yield False  # lock already held
    finally:
        if fd is not None:
            os.close(fd)
            try:
                os.unlink(LOCK_FILE)
            except FileNotFoundError:
                pass


def _cooldown_remaining(now: float) -> float:
    if _last_retrain_ts is None:
        return 0.0
    return max(0.0, RETRAIN_COOLDOWN_SECONDS - (now - _last_retrain_ts))


def _start_cooldown(now: float) -> None:
    global _last_retrain_ts
    _last_retrain_ts = now


retrain_runs_total = Counter("rul_retrain_runs_total", "Total number of retraining runs")
retrain_last_model_rmse_baseline = Gauge(
    "retrain_last_model_rmse_baseline", "Last trained model RMSE baseline on test."
)
retrain_last_training_fraction = Gauge(
    "retrain_last_training_fraction", "Last fraction of train_df used to train model."
)


def _check_mlflow_server(uri: str, timeout: int = 5) -> bool:
    import requests
    from requests.exceptions import RequestException, Timeout

    try:
        health_url = f"{uri.rstrip('/')}/health" if uri.startswith("http") else None
        if health_url:
            return requests.get(health_url, timeout=timeout).status_code == 200
        return False
    except (RequestException, Timeout):
        return False


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def _log_model_metadata(metadata: ModelMetadata) -> None:
    meta: dict[str, Any] = {
        "model_type": metadata.model_type,
        "feature_names": metadata.feature_names or [],
        "n_features": metadata.n_features,
        "target": metadata.target,
        "version": metadata.version,
        "test_rmse": metadata.test_rmse,
    }
    mlflow.log_params(
        {
            "metadata_model_type": metadata.model_type,
            "metadata_target": metadata.target,
            "metadata_version": metadata.version,
            "metadata_n_features": metadata.n_features,
            "metadata_test_rmse": metadata.test_rmse,
        }
    )
    mlflow.log_dict(meta, "model_metadata.json")


def _predict_and_log(model: Pipeline, feature_names: list[str], test_df: pd.DataFrame) -> Metrics:
    y_pred, y_test, metrics = eval_rul(model, test_df, feature_names)
    _ = plot_rmse(y_test, y_pred, metrics.rmse)
    mlflow.log_metric("test_rmse", metrics.rmse)
    mlflow.log_metric("test_r2", metrics.r2)
    mlflow.log_metric("test_mae", metrics.mae)
    mlflow.log_artifact(str(config.TEMP_FOLDER / "RUL_predictions_vs_actual.png"))
    return metrics


def _next_version(current: str | None) -> str:
    if not current:
        return "1.0"
    try:
        major, minor = current.split(".")
        return f"{int(major) + 1}.{int(minor)}"
    except Exception:
        return "1.0"


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(config.READY_DATA_PATH / "train.csv", index_col=False)
    test_df = pd.read_csv(config.READY_DATA_PATH / "test_last_rows.csv", index_col=False)
    return train_df, test_df


def _filter_by_fraction(
    train_df: pd.DataFrame, test_df: pd.DataFrame, fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Make the ordering explicit: smallest unit_numbers first
    unique_engines = np.sort(train_df["unit_number"].unique())

    num_engines = max(1, int(len(unique_engines) * fraction))
    selected = set(unique_engines[:num_engines])

    train_filtered = train_df[train_df["unit_number"].isin(selected)]
    test_filtered = test_df[test_df["unit_number"].isin(selected)]
    return train_filtered, test_filtered


def _run_retrain_job(model_name: str, model_path: Path, fraction: float | None) -> None:
    """
    Background job that performs retraining.
    All heavy work is done here so the API handler can 'fire and forget'.
    """
    now = time.monotonic()
    frac = fraction if fraction is not None else config.DEMO_FIRST_TRAIN_SIZE
    if frac <= 0 or frac > 1:
        print(f"[retrain-bg] invalid fraction={frac}, aborting", flush=True)
        return

    if not _check_mlflow_server(config.MLFLOW_TRACKING_URI):
        print(f"[retrain-bg] MLflow not accessible at {config.MLFLOW_TRACKING_URI}, aborting", flush=True)
        return

    if (rem := _cooldown_remaining(now)) > 0:
        print(f"[retrain-bg] cooldown active, remaining={rem:.1f}s, aborting", flush=True)
        return

    # try to acquire the lock; if not possible, just exit
    with try_acquire_train_lock() as got_lock:
        if not got_lock:
            print("[retrain-bg] training lock already held, aborting", flush=True)
            return

        print(f"[retrain-bg] starting, fraction={frac}", flush=True)
        model_file: Path = model_path / f"{model_name}.joblib"

        try:
            print("[retrain-bg] loading data", flush=True)
            train_df, test_df = _load_data()
            train_filtered, test_filtered = _filter_by_fraction(train_df, test_df, frac)
            feature_cols = _feature_columns(train_filtered)
            print(
                f"[retrain-bg] data ready: train={len(train_filtered)}, "
                f"test={len(test_filtered)}, n_features={len(feature_cols)}",
                flush=True,
            )

            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            mlflow.set_experiment("Random_Forest_Experiments")
            print("[retrain-bg] MLflow tracking/experiment set", flush=True)

            with mlflow.start_run():
                print("[retrain-bg] MLflow run started", flush=True)

                mlflow.set_tag("Model Type", "Random Forest")
                mlflow.set_tag("Task", "RUL Prediction")
                mlflow.set_tag("Data Processing", "Filtered by unit_number fraction")
                mlflow.log_param("train_samples", len(train_filtered))
                mlflow.log_param("features_count", len(feature_cols))
                mlflow.log_param("fraction", frac)

                print("[retrain-bg] calling fit_rf", flush=True)
                best_model, val_rmse = fit_rf(train_filtered)
                print(f"[retrain-bg] fit_rf done, val_rmse={val_rmse}", flush=True)
                mlflow.log_param("best_CV_params", best_model.get_params())
                mlflow.log_metric("validation_rmse", val_rmse if val_rmse is not None else -1)

                print("[retrain-bg] calling _predict_and_log", flush=True)
                metrics = _predict_and_log(best_model, feature_cols, test_filtered)
                print(f"[retrain-bg] _predict_and_log done, rmse={metrics.rmse}", flush=True)

                current_version = None
                try:
                    print("[retrain-bg] loading existing bundle for version", flush=True)
                    current_version = load_model_bundle(str(model_file)).metadata.version
                    print(f"[retrain-bg] current_version={current_version}", flush=True)
                except Exception as e:
                    print(f"[retrain-bg] could not load existing bundle (ok on first run): {e!r}", flush=True)

                new_version = _next_version(current_version)
                print(f"[retrain-bg] new_version={new_version}", flush=True)

                model_path.mkdir(parents=True, exist_ok=True)
                metadata = ModelMetadata(
                    model_type="RandomForestRegressor",
                    feature_names=feature_cols,
                    n_features=len(feature_cols),
                    target="RUL",
                    version=new_version,
                    test_rmse=metrics.rmse,
                )

                print("[retrain-bg] saving model bundle", flush=True)
                save_model_bundle(best_model, metadata, str(model_file))
                print("[retrain-bg] model bundle saved", flush=True)

                print("[retrain-bg] logging model metadata to MLflow", flush=True)
                _log_model_metadata(metadata)
                print("[retrain-bg] model metadata logged", flush=True)

                retrain_runs_total.inc()
                retrain_last_model_rmse_baseline.set(metrics.rmse)
                retrain_last_training_fraction.set(frac)

            print(f"-------------- Retraining completed v{new_version} --------------", flush=True)

            _start_cooldown(time.monotonic())

            del train_df, test_df, train_filtered, test_filtered, best_model
            gc.collect()

        except Exception as e:
            print(f"[retrain-bg] TOP-LEVEL EXCEPTION: {e!r}", flush=True)


@bentoml.service(
    workers=1,
    resources={"cpu": "4", "memory": "4Gi"},
    traffic={"timeout": 60, "concurrency": 10},
)
class RetrainingService:
    def __init__(self) -> None:
        self.model_name = "random_forest_model"
        self.model_path: Path = config.MODELS_PATH
        retrain_last_model_rmse_baseline.set(0.0)
        retrain_last_training_fraction.set(0.0)
        retrain_runs_total.inc()

    @bentoml.api
    def retrain(self, fraction: float | None = None) -> dict[str, Any]:
        """
        Fire-and-forget endpoint: schedule retraining in the background and return immediately.
        """
        # Optionally: cheap pre-checks before starting thread
        if fraction is not None and (fraction <= 0 or fraction > 1):
            return {"status": "error", "message": "fraction must be in (0,1]"}

        # quick cooldown check to avoid scheduling when obviously not allowed
        if (rem := _cooldown_remaining(time.monotonic())) > 0:
            return {
                "status": "error",
                "message": f"Retraining is cooling down. Try again in {int(rem)}s",
            }

        # Try acquiring the lock just to avoid scheduling duplicate jobs at the same instant.
        # We release it immediately; the background job will re-check and acquire for real.
        with try_acquire_train_lock() as got_lock:
            if not got_lock:
                return {
                    "status": "error",
                    "message": "Retraining already in progress. Try again in a few seconds.",
                }

        t = threading.Thread(
            target=_run_retrain_job,
            args=(self.model_name, self.model_path, fraction),
            daemon=True,
        )
        t.start()

        return {
            "status": "accepted",
            "message": "Retraining started in background.",
            "fraction": fraction if fraction is not None else config.DEMO_FIRST_TRAIN_SIZE,
        }

    @bentoml.api
    def get_baseline_rmse(self) -> float:
        return float(retrain_last_model_rmse_baseline._value.get())

    @bentoml.api
    def get_last_training_fraction(self) -> float:
        return float(retrain_last_training_fraction._value.get())


# OMP_NUM_THREADS=1
# OPENBLAS_NUM_THREADS=1
# MKL_NUM_THREADS=1
