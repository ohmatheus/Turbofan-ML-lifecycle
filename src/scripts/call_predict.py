import concurrent.futures
import time
from collections.abc import Hashable, Mapping
from typing import Any, Literal, Protocol, TypedDict, cast

import bentoml
import numpy as np
import pandas as pd
from datetime import datetime

from src.models.random_forest_utils import rmse_score
from src.utils.config import config

Payload = dict[str, list[dict[Hashable, Any]]]


class SuccessRequest(TypedDict):
    request_id: int
    success: Literal[True]
    duration: float
    result: Mapping[str, Any]


class FailedRequest(TypedDict):
    request_id: int
    success: Literal[False]
    duration: float
    error: str


Response = SuccessRequest | FailedRequest


class PredictClient(Protocol):
    def predict(self, data: "Payload") -> Mapping[str, Any]: ...
    def is_ready(self) -> bool: ...
    def close(self) -> None: ...


def make_single_request(client: PredictClient, data: Payload, request_id: int) -> Response:
    start_time = time.time()
    try:
        result = client.predict(data)
        end_time = time.time()
        return {"request_id": request_id, "success": True, "duration": end_time - start_time, "result": result}
    except Exception as e:
        end_time = time.time()
        return {"request_id": request_id, "success": False, "duration": end_time - start_time, "error": str(e)}


def concurrent_requests(num_requests: int = 10, max_workers: int = 5) -> None:
    test_last_rows: pd.DataFrame = pd.read_csv(config.READY_DATA_PATH / "test_last_rows.csv", index_col=False)
    data: Payload = {"rows": test_last_rows.to_dict("records")}

    client: PredictClient = bentoml.SyncHTTPClient("http://localhost:3000")

    if not client.is_ready():
        print("Service is not ready!")
        return

    print(f"Starting {num_requests} concurrent requests with {max_workers} workers...")
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: list[concurrent.futures.Future[Response]] = [
            executor.submit(make_single_request, client, data, i) for i in range(num_requests)
        ]

        results: list[Response] = []
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    total_time: float = time.time() - start_time

    successful_requests = cast("list[SuccessRequest]", [r for r in results if r["success"]])
    failed_requests = cast("list[FailedRequest]", [r for r in results if not r["success"]])

    print("\n=== Test Results ===")
    print(f"Total time: {total_time:.2f}s")
    print(f"Successful requests: {len(successful_requests)}/{num_requests}")
    print(f"Failed requests: {len(failed_requests)}")

    if count := len(successful_requests):
        durations: list[float] = [r["duration"] for r in successful_requests]
        print(f"Average response time: {sum(durations) / count:.3f}s")
        print(f"Min response time: {min(durations):.3f}s")
        print(f"Max response time: {max(durations):.3f}s")

    if failed_requests:
        print("\nErrors:")
        for req in failed_requests:
            print(f"  Request {req['request_id']}: {req['error']}")

    client.close()


def single_predict() -> None:
    test_last_rows: pd.DataFrame = pd.read_csv(config.READY_DATA_PATH / "test_last_rows.csv", index_col=False)
    data: Payload = {"rows": test_last_rows.to_dict("records")}
    client: PredictClient = bentoml.SyncHTTPClient("http://localhost:3000")

    if client.is_ready():
        try:
            result: Mapping[str, Any] = client.predict(data)
            # result = client.predict([3])

            y_test = test_last_rows["RUL"]
            y_pred = result.get("rul_predictions")
            rmse = rmse_score(np.asarray(y_test), np.asarray(y_pred))
            print(f"Test RMSE: {rmse:.2f}")

            print(
                f"Received {result['n_samples']} results with input shape: {result['input_shape']} with model version '{result['model_version']}'"
            )
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Prediction service is not ready")

    client.close()
    print("Client closed.")


def single_predict_with_feedback() -> None:
    test_last_rows = pd.read_csv(config.READY_DATA_PATH / "test_last_rows.csv", index_col=False)
    data: Payload = {"rows": test_last_rows.to_dict("records")}

    # Make prediction
    prediction_client = bentoml.SyncHTTPClient("http://localhost:3000")
    result = prediction_client.predict(data)

    # Extract results
    y_test = test_last_rows["RUL"]
    y_pred = result.get("rul_predictions")
    engine_ids = test_last_rows["unit_number"].astype(str)

    # Submit feedback for each prediction
    feedback_client = bentoml.SyncHTTPClient("http://localhost:3001")  # feedback service port

    for i, (actual, predicted, engine_id) in enumerate(zip(y_test, y_pred, engine_ids)):
        feedback_data = {
            "prediction_id": f"pred_{int(time.time())}_{i}",
            "predicted_rul": float(predicted),
            "actual_rul": float(actual),
            "engine_id": engine_id,
            "prediction_timestamp": datetime.now().isoformat(),
            "metadata": {
                "model_version": result.get("model_version"),
                "test_batch": True
            }
        }

        feedback_response = feedback_client.submit_feedback(feedback_data)
        print(f"Feedback submitted: {feedback_response}")

    # Calculate and display RMSE - just to ensure everything is ok
    rmse = rmse_score(np.asarray(y_test), np.asarray(y_pred))
    print(f"Test RMSE: {rmse:.2f}")

def main() -> None:
    print(f"TEST ENV : {config.TEST_ENV}")
    #single_predict()
    #concurrent_requests()
    single_predict_with_feedback()


if __name__ == "__main__":
    main()
#
#
