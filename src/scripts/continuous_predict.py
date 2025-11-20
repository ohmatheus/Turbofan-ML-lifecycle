import random
import threading
import time
from collections.abc import Hashable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import bentoml
import numpy as np
import pandas as pd

from src.models.random_forest_utils import rmse_score
from src.utils.config import config

Payload = dict[str, list[dict[Hashable, Any]]]


def simulate_user(user_id: int, test_data: pd.DataFrame, application_start: float, stop_event: threading.Event) -> None:
    prediction_client = bentoml.SyncHTTPClient("http://localhost:3000")
    feedback_client = bentoml.SyncHTTPClient("http://localhost:3001")

    request_count = 0

    try:
        while not stop_event.is_set():
            # Random wait between 0 and 5 second
            time.sleep(random.uniform(0, 5))

            current_time = time.time()
            # Compute how much of test_data is "unlocked" over time, from DEMO_FIRST_TRAIN_SIZE to 1.0
            elapsed_minutes = (current_time - application_start) / 60.0
            progress = min(1.0, max(0.0, elapsed_minutes / config.DEMO_DURATION_MINUTES))
            # fraction grows linearly from DEMO_FIRST_TRAIN_SIZE to 1.0
            available_fraction = config.DEMO_FIRST_TRAIN_SIZE + (1.0 - config.DEMO_FIRST_TRAIN_SIZE) * progress
            available_fraction = min(1.0, max(config.DEMO_FIRST_TRAIN_SIZE, available_fraction))

            total_rows = len(test_data)
            available_rows = max(1, int(total_rows * available_fraction))

            # Work only on the growing subset [0 : available_rows]
            available_data = test_data.iloc[:available_rows]

            print(f"User {user_id} - Available data: {len(available_data)} - fraction: {available_fraction:.2f}")

            # Select random rows from the currently available subset of test data
            num_rows = random.randint(1, min(config.PREDICTION_POOL_PER_USER, len(available_data)))
            selected_rows = available_data.sample(n=num_rows)

            request_count += 1
            request_id = f"user_{user_id}_req_{request_count}"

            # Make prediction
            try:
                data: Payload = {"rows": selected_rows.to_dict("records")}

                result: Mapping[str, Any] = {}
                if config.SIMULATE_ERRORS and random.randint(1, 10) == 1:
                    result = prediction_client.predict(
                        "some bad input"
                    )  # simulate 'bad_input' error, we could simulate other types of errors
                else:
                    result = prediction_client.predict(data)

                if result.get("error"):
                    print(f"User {user_id} - Request {request_count}: Error - {result.get('error')}")
                    continue

                # Submit feedback for each prediction
                y_actual = selected_rows["RUL"]
                y_pred = result.get("rul_predictions", [])
                rmse = rmse_score(np.asarray(y_actual), np.asarray(y_pred))
                engine_ids = selected_rows["unit_number"].astype(str)

                print(
                    f"User {user_id} - Request {request_count} - Test RMSE: {rmse:.2f}"  # this rmse is just for debug purpose
                )

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
                            "continuous_test": True,
                        },
                    }

                    feedback_response = feedback_client.submit_feedback(feedback_data)

                print(f"User {user_id} - Request {request_count}: Feedback submitted for {len(y_pred)} predictions.")
                print(
                    f"Feedback response {feedback_response['status']} - id :{feedback_response['feedback_id']} : {feedback_response['message']}"
                )

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

    prediction_client = bentoml.SyncHTTPClient("http://localhost:3000")
    feedback_client = bentoml.SyncHTTPClient("http://localhost:3001")
    drift_detection_client = bentoml.SyncHTTPClient("http://localhost:3003")

    all_ready = prediction_client.is_ready() and feedback_client.is_ready() and drift_detection_client.is_ready()

    # no need for them on this thread
    prediction_client.close()
    feedback_client.close()
    drift_detection_client.close()

    if not all_ready:
        print("Services are not ready!")
        return

    # Create NUM_USERS users
    num_workers = config.NUM_USERS
    print(f"Starting continuous prediction simulation with {num_workers} users.")
    print(
        f"Each user will select 1-{config.PREDICTION_POOL_PER_USER} random rows from {len(test_last_rows)} available test rows (increasing gradually)."
    )
    print("Press Ctrl+C to stop the simulation\n")

    stop_event = threading.Event()

    application_start: float = time.time()  # shared "timer start"

    # Start user simulation threads
    threads = []
    for user_id in range(num_workers):
        thread = threading.Thread(
            target=simulate_user, args=(user_id, test_last_rows, application_start, stop_event), daemon=True
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

        # Wait for all threads
        for thread in threads:
            thread.join(timeout=3)

        # delete_all_files(ask_retrain=False)
        print("All users stopped.")


def delete_all_files(ask_retrain: bool = True) -> None:
    print(f"TEST ENV : {config.TEST_ENV}")
    if not config.DELETE_MODEL_AT_DEMO_START:
        return

    model_file = config.MODELS_PATH / "random_forest_model.joblib"
    if model_file.exists():
        try:
            model_file.unlink()
            print(f"Deleted existing model file: {model_file}")
        except OSError as e:
            print(f"Failed to delete model file {model_file}: {e}")

    feedback_storage_path = Path(config.FEEDBACK_PATH / "rul_feedback.jsonl")
    if feedback_storage_path.exists():
        try:
            feedback_storage_path.unlink()
            print(f"Deleted feedback storage file: {feedback_storage_path}")
        except OSError as e:
            print(f"Failed to delete feedback storage file {feedback_storage_path}: {e}")

    if ask_retrain:
        retraining_client = bentoml.SyncHTTPClient("http://localhost:3004")
        try:
            if resp := retraining_client.retrain():
                print(f"Retrain trigger status: {resp.get('status')}")
        except Exception as e:
            print(f"Failed to trigger retrain: {e}")
        finally:
            retraining_client.close()


def main() -> None:
    delete_all_files()
    continuous_predict()


if __name__ == "__main__":
    main()
