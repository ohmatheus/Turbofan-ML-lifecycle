import bentoml
import pandas as pd

from src.utils.config import config


def predict() -> None:
    test_last_rows = pd.read_csv(config.READY_DATA_PATH / "test_last_rows.csv", index_col=False)
    data = {"rows": test_last_rows.to_dict("records")}
    client = bentoml.SyncHTTPClient("http://localhost:3000")

    if client.is_ready():
        try:
            result = client.predict(data)
            # result = client.predict([3])

            print(
                f"Received {result['n_samples']} results with input shape: {result['input_shape']} with model version '{result['model_version']}'"
            )
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Prediction service is not ready")

    client.close()
    print("Client closed.")


def main() -> None:
    print(f"TEST ENV : {config.TEST_ENV}")
    predict()


if __name__ == "__main__":
    main()
