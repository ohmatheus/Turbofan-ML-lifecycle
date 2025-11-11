from typing import Any, Union

import bentoml
import pandas as pd
import numpy as np
from bentoml.io import JSON
from pydantic import BaseModel
from src.models.model_bundle import ModelMetadata, load_model_bundle, ModelBundle
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
    type: str = None
    received_features: list[str] = None
    expected_features: list[str] = None
    missing_features: list[str] = None


@bentoml.service(
    resources={"cpu": "2", "memory": "2Gi"},
    traffic={"timeout": 60, "concurrency": 100, "max_concurrency": 200},
    monitoring={"enabled": True, "type": "default"}
)
class PredictionService:
    model_bundle: ModelBundle

    def __init__(self):
        print(f"TEST ENV : {config.TEST_ENV}")
        model_name = "full_sandbox"
        model_path = config.MODELS_PATH
        model_file = model_path / f"{model_name}.joblib"

        self.model_bundle = load_model_bundle(str(model_file))
        self.expected_features = self.model_bundle.get_n_features()

    @bentoml.api
    def predict(self, data: PredictionInput) -> dict[str, Any]:
        try:
            print(f"Received data type: {type(data)}")
            #print(f"Data content: {data}")

            samples = data.rows

            if not samples:
                return ErrorOutput(
                    error="No data provided",
                    type="validation_error"
                ).model_dump()

            df = pd.DataFrame(samples)

            print(f"Received {len(samples)} samples with {df.shape[1]} features")
            print(f"Expected features: {self.expected_features}")
            print(f"DataFrame shape: {df.shape}")

            # Get the expected feature names from model_bundle
            expected_features = set(self.model_bundle.get_feature_names())
            received_features = set(df.columns)

            # Check if all expected features are present
            missing_features = expected_features - received_features

            if missing_features:
                return ErrorOutput(
                    error=f"Missing required features: {sorted(missing_features)}",
                    type="validation_error",
                    received_features=list(df.columns),
                    expected_features=list(expected_features),
                    missing_features=list(missing_features)
                ).model_dump()

            predictions = self.model_bundle.model.predict(df[self.model_bundle.get_feature_names()])

            if isinstance(predictions, np.ndarray):
                predictions_list = predictions.tolist()
            else:
                predictions_list = [float(predictions)]

            return PredictionOutput(
                rul_predictions=predictions_list,
                model_version=self.model_bundle.metadata.version,
                n_samples=len(predictions_list),
                input_shape=list(df.shape)
            ).model_dump()

        except ValueError as e:
            return ErrorOutput(
                error=str(e),
                type="validation_error"
            ).model_dump()

        except Exception as e:
            return ErrorOutput(
                error=f"Internal server error: {str(e)}",
                type="internal_error"
            ).model_dump()

    #@bentoml.on_startup
    #@bentoml.on_deployement
    #@bentoml.on_shutdown
    #
    # /readyz
    # def __is_ready__(self) -> bool:
    #     # Check if required resources are available
    #     if self.db_connection is None or self.cache is None:
    #         return False
    #     return self.db_connection.is_connected() and self.cache.is_available()

    #__is_alive__ /livez
    #
    # import shutil
    # import bentoml
    #
    # local_model_dir = '/path/to/your/local/model/directory'
    #
    # with bentoml.models.create(
    #         name='my-local-model',  # Name of the model in the Model Store
    # ) as model_ref:
    #     # Copy the entire model directory to the BentoML Model Store
    #     shutil.copytree(local_model_dir, model_ref.path, dirs_exist_ok=True)
    #     print(f"Model saved: {model_ref}")
    #
    # import bentoml
    # from bentoml.models import BentoModel
    # import joblib
    #
    # @bentoml.service(resources={"cpu": "200m", "memory": "512Mi"})
    # class MyService:
    #     # Define model reference at the class level
    #     # Load a model from the Model Store or BentoCloud
    #     iris_ref = BentoModel("iris_sklearn:latest")
    #
    #     def __init__(self):
    #         self.iris_model = joblib.load(self.iris_ref.path_of("model.pkl"))
