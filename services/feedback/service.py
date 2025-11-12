import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import bentoml
import numpy as np
from prometheus_client import Counter, Histogram
from pydantic import BaseModel

from src.models.random_forest_utils import rmse_score
from src.utils.config import config


class FeedbackInput(BaseModel):
    prediction_id: str
    predicted_rul: float
    actual_rul: float
    engine_id: str
    prediction_timestamp: str
    feedback_timestamp: str | None = None
    metadata: dict[str, Any] = {}


class FeedbackResponse(BaseModel):
    status: str
    message: str
    feedback_id: str


feedback_counter = Counter("rul_feedback_total", "Total feedback entries received")
feedback_accuracy_histogram = Histogram("rul_prediction_accuracy", "Distribution of prediction errors")
feedback_rmse_histogram = Histogram("rul_rmse", "Distribution of rmse")


@bentoml.service(
    resources={"cpu": "1", "memory": "1Gi"},
    traffic={"timeout": 5, "concurrency": 50},
)
class FeedbackService:
    def __init__(self) -> None:
        self.feedback_storage_path = Path(config.FEEDBACK_PATH / "rul_feedback.jsonl")
        self.feedback_storage_path.parent.mkdir(parents=True, exist_ok=True)

    @bentoml.api
    def submit_feedback(self, feedback: FeedbackInput) -> FeedbackResponse:
        try:
            if feedback.feedback_timestamp is None:
                feedback.feedback_timestamp = datetime.now().isoformat()

            feedback_id = f"fb_{int(time.time() * 1000000)}"

            # Calculate prediction error for metrics
            error = abs(feedback.predicted_rul - feedback.actual_rul)
            feedback_accuracy_histogram.observe(error)

            # RMSE
            rmse = rmse_score(np.asarray([feedback.actual_rul]), np.asarray([feedback.predicted_rul]))
            print(f"Test RMSE: {rmse:.2f}")
            feedback_rmse_histogram.observe(rmse)

            feedback_record = {"feedback_id": feedback_id, **feedback.model_dump()}

            # Store feedback (JSONL file for simplicity)
            with open(self.feedback_storage_path, "a") as f:
                f.write(json.dumps(feedback_record) + "\n")

            feedback_counter.inc()

            return FeedbackResponse(status="success", message="Feedback recorded successfully", feedback_id=feedback_id)

        except Exception as e:
            return FeedbackResponse(status="error", message=f"Failed to record feedback: {str(e)}", feedback_id="")

    @bentoml.api
    def get_accuracy_summary(self) -> dict[str, Any]:
        try:
            feedback_data = []
            if self.feedback_storage_path.exists():
                with open(self.feedback_storage_path) as f:
                    feedback_data = [json.loads(line) for line in f]

            if not feedback_data:
                return {"message": "No feedback data available"}

            errors = [abs(fb["predicted_rul"] - fb["actual_rul"]) for fb in feedback_data]

            return {
                "total_feedback_entries": len(feedback_data),
                "mean_absolute_error": sum(errors) / len(errors),
                "max_error": max(errors),
                "min_error": min(errors),
                "accuracy_within_10_percent": sum(
                    1 for e, fb in zip(errors, feedback_data, strict=True) if e <= 0.1 * abs(fb["actual_rul"])
                )
                / len(feedback_data),
            }

        except Exception as e:
            return {"error": f"Failed to calculate accuracy: {str(e)}"}
