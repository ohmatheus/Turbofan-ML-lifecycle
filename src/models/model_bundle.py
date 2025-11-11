import joblib
from dataclasses import dataclass
from datetime import datetime
from typing import List
from sklearn.pipeline import Pipeline


@dataclass
class ModelMetadata:
    model_type: str = "RandomForestRegressor"
    feature_names: list[str] | None = None
    n_features: int = 0
    target: str = "rul"
    version: str = "0.0"


class ModelBundle:
    model: Pipeline
    metadata: ModelMetadata
    saved_at: str

    def __init__(self, model, metadata: ModelMetadata):
        self.model = model
        self.metadata = metadata
        self.saved_at = datetime.now().isoformat()

    def predict(self, x):
        return self.model.predict(x)

    def get_feature_names(self) -> List[str]:
        return self.metadata.feature_names

    def get_n_features(self) -> int:
        return self.metadata.n_features

    def validate_input_shape(self, x):
        expected_features = self.get_n_features()
        if expected_features and x.shape[1] != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {x.shape[1]}")


def save_model_bundle(model, metadata: ModelMetadata, filepath: str):
    bundle = ModelBundle(model, metadata)
    joblib.dump(bundle, filepath)
    print(f"Model bundle saved: {filepath}")


def load_model_bundle(filepath: str) -> ModelBundle:
    bundle = joblib.load(filepath)
    print(f"Model bundle loaded from: {filepath}")
    print(f"Model type: {bundle.metadata.model_type}")
    print(f"Version: {bundle.metadata.version}")
    print(f"Features: {bundle.get_n_features()}")
    return bundle


