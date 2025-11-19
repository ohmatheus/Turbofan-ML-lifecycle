import json
import time
from pathlib import Path
from typing import Any
import gc
from contextlib import contextmanager
from pathlib import Path
import os
import bentoml
import mlflow
import pandas as pd
import numpy as np
from prometheus_client import Counter, Gauge

from src.models.model_bundle import ModelMetadata, load_model_bundle, save_model_bundle
from src.models.random_forest_utils import EXCLUDE_COLS, Metrics, eval_rul, fit_rf, plot_rmse
from src.utils.config import config


RETRAIN_COOLDOWN_SECONDS = 30
_last_retrain_ts: float | None = None



LOCK_FILE = Path("/tmp/retrain.lock")

@contextmanager
def try_acquire_train_lock() -> bool:
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
retrain_baseline_rmse = Gauge("rul_retrain_baseline_rmse", "Baseline RMSE before retraining")
retrain_new_rmse = Gauge("rul_retrain_new_rmse", "Test RMSE after retraining")


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


def _predict_and_log(model, feature_names: list[str], test_df: pd.DataFrame) -> Metrics:
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


@bentoml.service(workers=4, resources={"cpu": "8", "memory": "8Gi"}, traffic={"timeout": 60, "concurrency": 10})
class RetrainingService:
    def __init__(self) -> None:
        self.model_name = "random_forest_model"
        self.model_path: Path = config.MODELS_PATH
        self.model_file: Path = self.model_path / f"{self.model_name}.joblib"

    @bentoml.api
    def retrain(self, fraction: float | None = None) -> dict[str, Any]:
        #----
        # import tracemalloc
        # tracemalloc.start()
        # # call retrain several times
        # snapshot = tracemalloc.take_snapshot()
        # for stat in snapshot.statistics('filename')[:20]:
        #     print(stat)
        #----

        now = time.monotonic()
        if (rem := _cooldown_remaining(now)) > 0:
            print("-------------- Ignoring --------------")
            return {"status": "error", "message": f"Retraining is cooling down. Try again in {int(rem)}s"}

        with try_acquire_train_lock() as got_lock:
            if not got_lock:
                return {
                    "status": "error",
                    "message": "Retraining already in progress. Try again in a few seconds.",
                }

        frac = fraction if fraction is not None else config.DEMO_FIRST_TRAIN_SIZE
        if frac <= 0 or frac > 1:
            return {"status": "error", "message": "fraction must be in (0,1]"}

        if not _check_mlflow_server(config.MLFLOW_TRACKING_URI):
            return {"status": "error", "message": f"MLflow not accessible at {config.MLFLOW_TRACKING_URI}"}

        print(f"[retrain] starting, fraction={frac}", flush=True)
        try:
            print("[retrain] loading data", flush=True)
            train_df, test_df = _load_data()
            train_filtered, test_filtered = _filter_by_fraction(train_df, test_df, frac)
            feature_cols = _feature_columns(train_filtered)
            print(
                f"[retrain] data ready: train={len(train_filtered)}, "
                f"test={len(test_filtered)}, n_features={len(feature_cols)}",
                flush=True,
            )

            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            mlflow.set_experiment("Random_Forest_Experiments")
            print("[retrain] MLflow tracking/experiment set", flush=True)

            with mlflow.start_run():
                print("[retrain] MLflow run started", flush=True)

                mlflow.set_tag("Model Type", "Random Forest")
                mlflow.set_tag("Task", "RUL Prediction")
                mlflow.set_tag("Data Processing", "Filtered by unit_number fraction")
                mlflow.log_param("train_samples", len(train_filtered))
                mlflow.log_param("features_count", len(feature_cols))
                mlflow.log_param("fraction", frac)

                print("[retrain] calling fit_rf", flush=True)
                best_model, val_rmse = fit_rf(train_filtered)
                print(f"[retrain] fit_rf done, val_rmse={val_rmse}", flush=True)
                mlflow.log_param("best_CV_params", best_model.get_params())
                mlflow.log_metric("validation_rmse", val_rmse if val_rmse is not None else -1)

                print("[retrain] calling _predict_and_log", flush=True)
                metrics = _predict_and_log(best_model, feature_cols, test_filtered)
                print(f"[retrain] _predict_and_log done, rmse={metrics.rmse}", flush=True)

                current_version = None
                if fraction is not None:
                    try:
                        print("[retrain] loading existing bundle for version", flush=True)
                        current_version = load_model_bundle(str(self.model_file)).metadata.version
                        print(f"[retrain] current_version={current_version}", flush=True)
                    except Exception as e:
                        print(f"[retrain] could not load existing bundle (ok on first run): {e!r}", flush=True)

                new_version = _next_version(current_version)
                print(f"[retrain] new_version={new_version}", flush=True)

                self.model_path.mkdir(parents=True, exist_ok=True)
                metadata = ModelMetadata(
                    model_type="RandomForestRegressor",
                    feature_names=feature_cols,
                    n_features=len(feature_cols),
                    target="RUL",
                    version=new_version,
                    test_rmse=metrics.rmse,
                )

                print("[retrain] saving model bundle", flush=True)
                save_model_bundle(best_model, metadata, str(self.model_file))
                print("[retrain] model bundle saved", flush=True)

                print("[retrain] logging model metadata to MLflow", flush=True)
                _log_model_metadata(metadata)
                print("[retrain] model metadata logged", flush=True)

                retrain_runs_total.inc()
                if metrics.rmse is not None:
                    retrain_baseline_rmse.set(metrics.rmse)
                    mlflow.log_metric("baseline_rmse", metrics.rmse)
                retrain_new_rmse.set(metrics.rmse)

            print(f"-------------- Retraining completed v{new_version} --------------", flush=True)

            now = time.monotonic()
            _start_cooldown(now)

            del train_df, test_df, train_filtered, test_filtered, best_model
            gc.collect()

            return {
                "status": "success",
                "message": "Retraining completed",
                "model_name": self.model_name,
                "version": new_version,
                "fraction": frac,
                "metrics": {
                    "rmse": metrics.rmse,
                    "r2": metrics.r2,
                    "mae": metrics.mae,
                },
            }

        except Exception as e:
            # One catch for the whole flow; MLflow will still try to mark the run FAILED
            print(f"[retrain] TOP-LEVEL EXCEPTION: {e!r}", flush=True)
            return {
                "status": "error",
                "message": f"Retraining failed: {e}",
            }

#OMP_NUM_THREADS=1
#OPENBLAS_NUM_THREADS=1
#MKL_NUM_THREADS=1
