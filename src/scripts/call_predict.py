from typing import Any

import bentoml
import pandas as pd
from pydantic import BaseModel

from src.utils.config import config


class PredictionInput(BaseModel):
    rows: list[dict[str, Any]]


def predict() -> None:
    test_last_rows = pd.read_csv(config.READY_DATA_PATH / "test_last_rows.csv", index_col=False)

    data = {"rows": test_last_rows.to_dict("records")}

    client = bentoml.SyncHTTPClient("http://localhost:3000")

    result = None
    if client.is_ready():
        result = client.predict(data)
    else:
        print("Prediction service is not ready")

    client.close()

    print(
        f"Received {result['n_samples']} results with input shape: {result['input_shape']} with model version '{result['model_version']}'"
    )
    # feature_cols = [col for col in test_last_rows.columns if col not in EXCLUDE_COLS]
    # x_test = test_last_rows[feature_cols]
    #
    #
    # _ = plot_rmse(y_test, y_pred, metrics.rmse)
    #
    # print(f"Eval completed. Test RMSE: {metrics.rmse:.4f}")


def main() -> None:
    print(f"TEST ENV : {config.TEST_ENV}")

    predict()


if __name__ == "__main__":
    main()
