import concurrent.futures
import random
import time
from collections.abc import Hashable, Mapping
from datetime import datetime
from typing import Any, Literal, Protocol, TypedDict, cast

import bentoml
import numpy as np
import pandas as pd

from src.models.random_forest_utils import rmse_score
from src.utils.config import config

Payload = dict[str, list[dict[Hashable, Any]]]


def simulate_user(user_id: int, test_data: pd.DataFrame, stop_event) -> None:
    prediction_client = bentoml.SyncHTTPClient("http://localhost:3000")
    feedback_client = bentoml.SyncHTTPClient("http://localhost:3001")

    request_count = 0

    try:
        while not stop_event.is_set():
            # Random wait between 0 and 1 second
            time.sleep(random.uniform(0, 1))

            # Select random rows from test data
            num_rows = random.randint(1, min(config.CONTINUOUS_PREDICT_RANGE, len(test_data)))
            selected_rows = test_data.sample(n=num_rows)

            request_count += 1
            request_id = f"user_{user_id}_req_{request_count}"

            # Make prediction
            try:
                start_time = time.time()
                data: Payload = {"rows": selected_rows.to_dict("records")}
                result = prediction_client.predict(data)
                prediction_time = time.time() - start_time


                # Submit feedback for each prediction
                y_actual = selected_rows["RUL"]
                y_pred = result.get("rul_predictions", [])
                rmse = rmse_score(np.asarray(y_actual), np.asarray(y_pred))
                engine_ids = selected_rows["unit_number"].astype(str)

                print(f"User {user_id} - Request {request_count}: Prediction successful ({prediction_time:.3f}s) - Test RMSE: {rmse:.2f}")

                feedback_start = time.time()
                for i, (actual, predicted, engine_id) in enumerate(zip(y_actual, y_pred, engine_ids, strict=True)):
                    feedback_data = {
                        "prediction_id": f"{request_id}_pred_{i}",
                        "predicted_rul": float(predicted),
                        "actual_rul": float(actual),
                        "engine_id": engine_id,
                        "prediction_timestamp": datetime.now().isoformat(),
                        "metadata": {
                            "model_version": result.get("model_version"),
                            "user_id": user_id,
                            "request_id": request_id,
                            "continuous_test": True
                        },
                    }

                    feedback_response = feedback_client.submit_feedback(feedback_data)

                feedback_time = time.time() - feedback_start
                total_time = prediction_time + feedback_time

                print(
                    f"User {user_id} - Request {request_count}: Feedback submitted for {len(y_pred)} predictions ({feedback_time:.3f}s) - Total: {total_time:.3f}s")

            except Exception as e:
                print(f"User {user_id} - Request {request_count}: Error - {str(e)}")

    except KeyboardInterrupt:
        print(f"User {user_id} stopped by keyboard interrupt")
    finally:
        prediction_client.close()
        feedback_client.close()
        print(f"User {user_id} completed {request_count} requests")


def continuous_predict() -> None:
    """Run continuous prediction simulation with multiple users"""
    test_last_rows = pd.read_csv(config.READY_DATA_PATH / "test_last_rows.csv", index_col=False)

    # Test if services are ready
    prediction_client = bentoml.SyncHTTPClient("http://localhost:3000")
    feedback_client = bentoml.SyncHTTPClient("http://localhost:3001")

    if not prediction_client.is_ready():
        print("Prediction service is not ready!")
        prediction_client.close()
        return

    if not feedback_client.is_ready():
        print("Feedback service is not ready!")
        feedback_client.close()
        return

    prediction_client.close()
    feedback_client.close()

    # Create 5 users
    num_workers = 5
    print(f"Starting continuous prediction simulation with {num_workers} users")
    print(
        f"Each user will select 1-{config.CONTINUOUS_PREDICT_RANGE} random rows from {len(test_last_rows)} available test rows")
    print("Press Ctrl+C to stop the simulation\n")

    # Create a stop event for graceful shutdown
    import threading
    stop_event = threading.Event()

    # Start user simulation threads
    threads = []
    for user_id in range(num_workers):
        thread = threading.Thread(
            target=simulate_user,
            args=(user_id, test_last_rows, stop_event),
            daemon=True
        )
        thread.start()
        threads.append(thread)
        print(f"Started user {user_id}")

    try:
        # Keep the main thread alive and wait for keyboard interrupt
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping simulation...")
        stop_event.set()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=2)  # Wait up to 5 seconds for each thread

        print("All users stopped.")


def main() -> None:
    print(f"TEST ENV : {config.TEST_ENV}")
    continuous_predict()


if __name__ == "__main__":
    main()