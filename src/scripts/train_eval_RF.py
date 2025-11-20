import mlflow
import pandas as pd
import requests
from requests.exceptions import RequestException, Timeout
from sklearn.pipeline import Pipeline

from src.models.model_bundle import ModelMetadata, save_model_bundle
from src.models.random_forest_utils import EXCLUDE_COLS, Metrics, eval_rul, fit_rf, plot_rmse
from src.utils.config import config

MODEL_VERSION = "1.0"
MODEL_NAME = "random_forest_model"


def log_model_metadata(metadata: ModelMetadata) -> None:
    meta = {
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


def check_mlflow_server(uri: str, timeout: int = 5) -> bool:
    try:
        health_url = f"{uri.rstrip('/')}/health" if uri.startswith("http") else None
        if health_url:
            response = requests.get(health_url, timeout=timeout)
            return bool(response.status_code == 200)
        return False
    except (RequestException, Timeout):
        return False


# def load_model_and_predict(model_name: str) -> None:
#     model_path = config.MODELS_PATH
#     model_file = model_path / f"{model_name}.joblib"
#     bundle = load_model_bundle(str(model_file))
#
#     test_df = pd.read_csv(config.READY_DATA_PATH / "test_last_rows.csv", index_col=False)
#
#     feature_names = bundle.get_feature_names()
#     if feature_names is None:
#         print("Feature names not found in model bundle. Skipping prediction.")
#         return
#     y_pred, y_test, metrics = eval_rul(bundle.model, test_df, feature_names)
#     _ = plot_rmse(y_test, y_pred, metrics.rmse)
#
#     print(f"Eval completed. Test RMSE: {metrics.rmse:.4f}")
#     mlflow.log_metric("test_rmse", metrics.rmse)
#     mlflow.log_metric("test_r2", metrics.r2)
#     mlflow.log_metric("test_mae", metrics.mae)
#     mlflow.log_artifact(str(config.TEMP_FOLDER / "RUL_predictions_vs_actual.png"))


def predict(model: Pipeline, feature_names: list[str], test_df: pd.DataFrame) -> Metrics:
    y_pred, y_test, metrics = eval_rul(model, test_df, feature_names)
    _ = plot_rmse(y_test, y_pred, metrics.rmse)

    print(f"Eval completed. Test RMSE: {metrics.rmse:.4f}")
    mlflow.log_metric("test_rmse", metrics.rmse)
    mlflow.log_metric("test_r2", metrics.r2)
    mlflow.log_metric("test_mae", metrics.mae)
    mlflow.log_artifact(str(config.TEMP_FOLDER / "RUL_predictions_vs_actual.png"))
    return metrics


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
    test_df = pd.read_csv(config.READY_DATA_PATH / "test_last_rows.csv", index_col=False)

    unique_engines = train_df["unit_number"].unique()
    # Take first X% of engine units `unit_number`
    num_engines = int(len(unique_engines) * config.DEMO_FIRST_TRAIN_SIZE)
    selected_engines = unique_engines[:num_engines]

    # Filter to get all rows for those unit_number
    train_filtered = train_df[train_df["unit_number"].isin(selected_engines)]
    test_filtered = test_df[test_df["unit_number"].isin(selected_engines)]
    feature_cols = [col for col in train_filtered.columns if col not in EXCLUDE_COLS]

    # mlflow.end_run()
    with mlflow.start_run(nested=True):
        mlflow.set_tag("Model Type", "Random Forest")
        mlflow.set_tag("Task", "RUL Prediction")
        mlflow.set_tag("Data Processing", "Filtered RUL thresholds applied")

        mlflow.log_param("train_samples", len(train_filtered))
        mlflow.log_param("features_count", len(feature_cols))

        best_model, val_rmse = fit_rf(train_filtered)

        mlflow.log_param("best_CV_params", best_model.get_params())

        val_rmse = val_rmse if val_rmse is not None else -1
        mlflow.log_metric("validation_rmse", val_rmse)

        metrics = predict(best_model, feature_cols, test_filtered)

        print(f"Test RMSE: {metrics.rmse:.4f}")

        model_name = MODEL_NAME
        mlflow.log_param("model_version", model_name)
        model_path = config.MODELS_PATH
        model_path.mkdir(exist_ok=True)
        model_file = model_path / f"{model_name}.joblib"

        metadata = ModelMetadata(
            model_type="RandomForestRegressor",
            feature_names=feature_cols,
            n_features=len(feature_cols),
            target="RUL",
            version=MODEL_VERSION,
            test_rmse=metrics.rmse,
            # nb_first_rows_used_for_training: int = 0
        )
        save_model_bundle(best_model, metadata, str(model_file))
        log_model_metadata(metadata)

        print(f"MLflow run completed. Validation RMSE: {val_rmse:.4f}")


if __name__ == "__main__":
    main()
