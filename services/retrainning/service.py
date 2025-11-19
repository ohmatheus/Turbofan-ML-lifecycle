import json
from pathlib import Path
from typing import Any

import bentoml
import mlflow
import pandas as pd
from prometheus_client import Counter, Gauge

from src.models.model_bundle import ModelMetadata, load_model_bundle, save_model_bundle
from src.models.random_forest_utils import EXCLUDE_COLS, Metrics, eval_rul, fit_rf, plot_rmse
from src.utils.config import config


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
        return f"{int(major)}.{int(minor) + 1}"
    except Exception:
        return "1.0"


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(config.READY_DATA_PATH / "train.csv", index_col=False)
    test_df = pd.read_csv(config.READY_DATA_PATH / "test_last_rows.csv", index_col=False)
    return train_df, test_df


def _filter_by_fraction(train_df: pd.DataFrame, test_df: pd.DataFrame, fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_engines = train_df["unit_number"].unique()
    num_engines = max(1, int(len(unique_engines) * fraction))
    selected = set(unique_engines[:num_engines])
    train_filtered = train_df[train_df["unit_number"].isin(selected)]
    test_filtered = test_df[test_df["unit_number"].isin(selected)]
    return train_filtered, test_filtered


@bentoml.service(resources={"cpu": "4", "memory": "4Gi"}, traffic={"timeout": 10, "concurrency": 1})
class RetrainingService:
    def __init__(self) -> None:
        self.model_name = "random_forest_model"
        self.model_path: Path = config.MODELS_PATH
        self.model_file: Path = self.model_path / f"{self.model_name}.joblib"

    @bentoml.api
    def retrain(self, fraction: float | None = None) -> dict[str, Any]:
        print("Retraining service called")
        frac = fraction if fraction is not None else config.DEMO_FIRST_TRAIN_SIZE
        if frac <= 0 or frac > 1:
            return {"status": "error", "message": "fraction must be in (0,1]"}

        if not _check_mlflow_server(config.MLFLOW_TRACKING_URI):
            return {"status": "error", "message": f"MLflow not accessible at {config.MLFLOW_TRACKING_URI}"}

        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment("Random_Forest_Experiments")

        train_df, test_df = _load_data()
        train_filtered, test_filtered = _filter_by_fraction(train_df, test_df, frac)
        feature_cols = _feature_columns(train_filtered)

        baseline_rmse: float | None = None
        if self.model_file.exists():
            try:
                bundle = load_model_bundle(str(self.model_file))
                feats = bundle.metadata.feature_names or feature_cols
                _, _, base_metrics = eval_rul(bundle.model, test_filtered, feats)
                baseline_rmse = float(base_metrics.rmse)
            except Exception:
                baseline_rmse = None

        with mlflow.start_run(nested=True):
            mlflow.set_tag("Model Type", "Random Forest")
            mlflow.set_tag("Task", "RUL Prediction")
            mlflow.set_tag("Data Processing", "Filtered by unit_number fraction")
            mlflow.log_param("train_samples", len(train_filtered))
            mlflow.log_param("features_count", len(feature_cols))
            mlflow.log_param("fraction", frac)

            best_model, val_rmse = fit_rf(train_filtered)
            mlflow.log_param("best_CV_params", best_model.get_params())
            mlflow.log_metric("validation_rmse", val_rmse if val_rmse is not None else -1)

            metrics = _predict_and_log(best_model, feature_cols, test_filtered)

            current_version = None
            if self.model_file.exists():
                try:
                    current_version = load_model_bundle(str(self.model_file)).metadata.version
                except Exception:
                    current_version = None
            new_version = _next_version(current_version)

            self.model_path.mkdir(parents=True, exist_ok=True)
            metadata = ModelMetadata(
                model_type="RandomForestRegressor",
                feature_names=feature_cols,
                n_features=len(feature_cols),
                target="RUL",
                version=new_version,
                test_rmse=metrics.rmse,
            )
            save_model_bundle(best_model, metadata, str(self.model_file))
            _log_model_metadata(metadata)

            retrain_runs_total.inc()
            if baseline_rmse is not None:
                retrain_baseline_rmse.set(baseline_rmse)
                mlflow.log_metric("baseline_rmse", baseline_rmse)
            retrain_new_rmse.set(metrics.rmse)

            return {
                "status": "success",
                "message": "Retraining completed",
                "model_name": self.model_name,
                "version": new_version,
                "fraction": frac,
                "metrics": {
                    "baseline_rmse": baseline_rmse,
                    "rmse": metrics.rmse,
                    "r2": metrics.r2,
                    "mae": metrics.mae,
                },
            }
