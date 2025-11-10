import joblib
import mlflow
import pandas as pd
import requests
from requests.exceptions import RequestException, Timeout

from src.models.random_forest_utils import eval_rul, fit_rf, plot_rmse
from src.utils.config import config


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
    model = joblib.load(model_file)

    test_df = pd.read_csv(config.READY_DATA_PATH / "test.csv", index_col=False)

    y_pred, y_test, metrics = eval_rul(model, test_df)
    _ = plot_rmse(y_test, y_pred, metrics.rmse)

    print(f"Eval completed. Validation RMSE: {metrics.rmse:.4f}")
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
        joblib.dump(best_model, model_file)
        print(f"Model saved to: {model_file}")

        print(f"MLflow run completed. Validation RMSE: {val_rmse:.4f}")

        load_model_and_predict(model_name)


if __name__ == "__main__":
    main()
