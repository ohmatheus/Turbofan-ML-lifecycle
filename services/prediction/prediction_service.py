import threading
import time
from typing import Any

import bentoml
import numpy as np
import pandas as pd
from prometheus_client import Counter, Histogram
from pydantic import BaseModel
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from src.models.model_bundle import ModelBundle, load_model_bundle
from src.utils.config import config


class PredictionInput(BaseModel):
    rows: list[dict[str, Any]]


class PredictionOutput(BaseModel):
    rul_predictions: list[float]
    model_version: str
    n_samples: int
    input_shape: list[int]


class ErrorOutput(BaseModel):
    error: str
    type: str | None = None
    received_features: list[str] | None = None
    expected_features: list[str] | None = None
    missing_features: list[str] | None = None


prediction_counter = Counter(
        name="rul_predictions_total",
        documentation="Total number of RUL predictions made",
        labelnames=["status", "model_version"],
)
prediction_time_histogram = Histogram(
    name="rul_prediction_duration_seconds",
    documentation="Time spent on RUL predictions",
    labelnames=["model_version"],
)
error_counter = Counter(
    name="rul_prediction_errors_total",
    documentation="Total number of prediction errors",
    labelnames=["error_type"]
)
model_reload_counter = Counter(
    name="model_reloads_total",
    documentation="Total number of model reloads",
    labelnames=["status"]
)


class ModelReloadHandler(FileSystemEventHandler):
    def __init__(self, service_instance: "PredictionService") -> None:
        self.service = service_instance

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return

        src_path = event.src_path
        if isinstance(src_path, bytes):
            src_path = src_path.decode("utf-8")

        if src_path.endswith(".joblib"):
            print(f"Detected model file change: {src_path}")
            self.service.reload_model()


@bentoml.service(
    resources={"cpu": "2", "memory": "2Gi"},
    traffic={"timeout": 5, "concurrency": 100, "max_concurrency": 200},
)
class PredictionService:
    model_bundle: ModelBundle
    observer: BaseObserver | None = None

    def __init__(self) -> None:
        print(f"TEST ENV : {config.TEST_ENV}")
        self.model_name = "full_sandbox"
        self.model_path = config.MODELS_PATH
        self.model_file = self.model_path / f"{self.model_name}.joblib"
        self._model_lock = threading.RLock()
        self._load_model()
        self._setup_file_watcher()

    def _load_model(self) -> None:
        try:
            self.model_bundle = load_model_bundle(str(self.model_file))
            self.expected_features = self.model_bundle.get_n_features()
            print(f"Model loaded: version {self.model_bundle.metadata.version}")
            #model_reload_counter.labels(status="success").inc()
        except Exception as e:
            print(f"Failed to load model: {e}")
            #model_reload_counter.labels(status="error").inc()
            raise

    def _setup_file_watcher(self) -> None:
        try:
            self.observer = Observer()
            event_handler = ModelReloadHandler(self)
            self.observer.schedule(event_handler, str(self.model_path), recursive=False)
            self.observer.start()
            print(f"File watcher started for {self.model_path}")
        except ImportError:
            print("watchdog not installed. Hot-reload disabled.")
            self.observer = None
        except Exception as e:
            print(f"Failed to setup file watcher: {e}")
            self.observer = None

    def reload_model(self) -> None:
        with self._model_lock:
            print("Reloading model...")
            try:
                # Small delay to ensure file write is complete
                time.sleep(1)
                self._load_model()
                print("Model reloaded successfully")
            except Exception as e:
                print(f"Failed to reload model: {e}")

    @bentoml.api
    def predict(self, data: PredictionInput) -> dict[str, Any]:
        start_time = time.time()
        try:
            print(f"Received data type: {type(data)}")
            # print(f"Data content: {data}")

            samples = data.rows

            if not samples:
                return ErrorOutput(error="No data provided", type="validation_error").model_dump()

            df = pd.DataFrame(samples)

            print(f"Received {len(samples)} samples with {df.shape[1]} features")
            print(f"Expected features: {self.expected_features}")
            print(f"DataFrame shape: {df.shape}")

            # Get the expected feature names from model_bundle
            feature_names = self.model_bundle.get_feature_names()
            if feature_names is None:
                return ErrorOutput(error="Missing feature names", type="validation_error").model_dump()

            expected_features = set(feature_names)
            received_features = set(df.columns)

            # Check if all expected features are present
            missing_features = expected_features - received_features

            if missing_features:
                #error_counter.labels(error_type="missing_features").inc()
                return ErrorOutput(
                    error=f"Missing required features: {sorted(missing_features)}",
                    type="validation_error",
                    received_features=list(df.columns),
                    expected_features=list(expected_features),
                    missing_features=list(missing_features),
                ).model_dump()

            predictions = self.model_bundle.model.predict(df[self.model_bundle.get_feature_names()])

            if isinstance(predictions, np.ndarray):
                predictions_list = predictions.tolist()
            else:
                predictions_list = [float(predictions)]

            # Record metrics
            prediction_counter.labels(status="success", model_version=self.model_bundle.metadata.version).inc()
            prediction_time_histogram.labels(model_version=self.model_bundle.metadata.version).observe(
                time.time() - start_time
            )

            return PredictionOutput(
                rul_predictions=predictions_list,
                model_version=self.model_bundle.metadata.version,
                n_samples=len(predictions_list),
                input_shape=list(df.shape),
            ).model_dump()

        except ValueError as e:
            error_counter.labels(error_type="validation_error").inc()
            return ErrorOutput(error=str(e), type="validation_error").model_dump()

        except Exception as e:
            error_counter.labels(error_type="internal_error").inc()
            return ErrorOutput(error=f"Internal server error: {str(e)}", type="internal_error").model_dump()

    @bentoml.api
    def reload(self) -> dict[str, str]:
        try:
            self.reload_model()
            return {"status": "success", "message": "Model reloaded successfully"}
        except Exception as e:
            return {"status": "error", "message": f"Failed to reload model: {str(e)}"}

    @bentoml.on_shutdown
    def cleanup(self) -> None:
        if self.observer:
            self.observer.stop()
            self.observer.join()
