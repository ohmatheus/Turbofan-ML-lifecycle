from dataclasses import dataclass
from datetime import datetime

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


@dataclass
class ModelMetadata:
    model_type: str = "RandomForestRegressor"
    feature_names: list[str] | None = None
    n_features: int = 0
    target: str = "rul"
    version: str = "0.0"
    test_rmse: float = 0.0
    # nb_first_rows_used_for_training: int = 0


class ModelBundle:
    model: Pipeline
    metadata: ModelMetadata
    saved_at: str

    def __init__(self, model: Pipeline, metadata: ModelMetadata) -> None:
        self.model = model
        self.metadata = metadata
        self.saved_at = datetime.now().isoformat()

    def get_feature_names(self) -> list[str] | None:
        return self.metadata.feature_names

    def get_n_features(self) -> int:
        return self.metadata.n_features

    def validate_input_shape(self, x: pd.DataFrame) -> None:
        expected_features = self.get_n_features()
        if expected_features and x.shape[1] != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {x.shape[1]}")


def save_model_bundle(model: Pipeline, metadata: ModelMetadata, filepath: str) -> None:
    bundle = ModelBundle(model, metadata)
    joblib.dump(bundle, filepath)
    print(f"Model bundle saved: {filepath}")


def load_model_bundle(filepath: str) -> ModelBundle:
    try:
        bundle: ModelBundle = joblib.load(filepath)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model file not found: {filepath}") from e
    print(f"Model bundle loaded from: {filepath}")
    print(f"Model type: {bundle.metadata.model_type}")
    print(f"Version: {bundle.metadata.version}")
    print(f"Features: {bundle.get_n_features()}")
    return bundle
