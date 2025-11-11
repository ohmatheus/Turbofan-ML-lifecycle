import mlflow
import pandas as pd
import requests
from requests.exceptions import RequestException, Timeout

from src.models.model_bundle import ModelMetadata, load_model_bundle, save_model_bundle
from src.models.random_forest_utils import EXCLUDE_COLS, eval_rul, fit_rf, plot_rmse
from src.utils.config import config


def log_model_metadata(metadata: ModelMetadata) -> None:
    meta = {
        "model_type": metadata.model_type,
        "feature_names": metadata.feature_names or [],
        "n_features": metadata.n_features,
        "target": metadata.target,
        "version": metadata.version,
    }
    mlflow.log_params(
        {
            "metadata_model_type": metadata.model_type,
            "metadata_target": metadata.target,
            "metadata_version": metadata.version,
            "metadata_n_features": metadata.n_features,
        }
    )
    mlflow.log_dict(meta, "model_metadata.json")


def check_mlflow_server(uri: str, timeout: int = 5) -> bool:
    try:
        health_url = f"{uri.rstrip('/')}/health" if uri.startswith("http") else None
        if health_url:
            response = requests.get(health_url, timeout=timeout)
            return bool(response.status_code == 200)
        return False
    except (RequestException, Timeout):
        return False


def load_model_and_predict(model_name: str) -> None:
    model_path = config.MODELS_PATH
    model_file = model_path / f"{model_name}.joblib"
    bundle = load_model_bundle(str(model_file))

    test_df = pd.read_csv(config.READY_DATA_PATH / "test_last_rows.csv", index_col=False)

    feature_names = bundle.get_feature_names()
    if feature_names is None:
        print("Feature names not found in model bundle. Skipping prediction.")
        return
    y_pred, y_test, metrics = eval_rul(bundle.model, test_df, feature_names)
    _ = plot_rmse(y_test, y_pred, metrics.rmse)

    print(f"Eval completed. Test RMSE: {metrics.rmse:.4f}")
    mlflow.log_metric("test_rmse", metrics.rmse)
    mlflow.log_metric("test_r2", metrics.r2)
    mlflow.log_metric("test_mae", metrics.mae)
    mlflow.log_artifact(str(config.TEMP_FOLDER / "RUL_predictions_vs_actual.png"))


def main() -> None:
    if check_mlflow_server(config.MLFLOW_TRACKING_URI):
        try:
            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            print(f"Connected to MLflow server at {config.MLFLOW_TRACKING_URI}")
        except Exception as e:
            print(f"Error setting MLflow tracking URI: {e}")
            print("Using local MLflow tracking")
            return
    else:
        print(f"MLflow server not accessible at {config.MLFLOW_TRACKING_URI}. Start local MLFlow server first !!")
        return

    experiment_name = "Random_Forest_Experiments"
    mlflow.set_experiment(experiment_name)

    train_df = pd.read_csv(config.READY_DATA_PATH / "train.csv", index_col=False)

    with mlflow.start_run():
        mlflow.set_tag("Model Type", "Random Forest")
        mlflow.set_tag("Task", "RUL Prediction")
        mlflow.set_tag("Data Processing", "Filtered RUL thresholds applied")

        mlflow.log_param("train_samples", len(train_df))
        mlflow.log_param("features_count", len(train_df.columns) - 1)

        best_model, val_rmse = fit_rf(train_df)

        mlflow.log_param("best_CV_params", best_model.get_params())

        val_rmse = val_rmse if val_rmse is not None else -1
        mlflow.log_metric("validation_rmse", val_rmse)

        model_name = "full_sandbox"
        mlflow.log_param("model_version", model_name)
        model_path = config.MODELS_PATH
        model_path.mkdir(exist_ok=True)
        model_file = model_path / f"{model_name}.joblib"

        feature_cols = [col for col in train_df.columns if col not in EXCLUDE_COLS]
        metadata = ModelMetadata(
            model_type="RandomForestRegressor",
            feature_names=feature_cols,
            n_features=len(feature_cols),
            target="RUL",
            version=model_name,
        )
        save_model_bundle(best_model, metadata, str(model_file))
        log_model_metadata(metadata)

        print(f"MLflow run completed. Validation RMSE: {val_rmse:.4f}")

        load_model_and_predict(model_name)


if __name__ == "__main__":
    main()
