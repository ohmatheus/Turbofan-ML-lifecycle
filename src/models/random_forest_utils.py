import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from src.utils.config import config

EXCLUDE_COLS = ["subset", "unit_number", "time_cycles", "RUL"]


@dataclass
class Metrics:
    rmse: float
    r2: float
    mae: float


pipeline = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        (
            "rf",
            RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=46,
                n_jobs=-1,
            ),
        ),
    ]
)


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


rmse_scorer = make_scorer(rmse_score, greater_is_better=False)

param_grid = {
    "rf__n_estimators": [300, 500, 700],
    "rf__max_depth": [None, 10, 20, 30],
    "rf__min_samples_split": [2, 5, 10],
    "rf__min_samples_leaf": [1, 2, 4],
    "rf__max_features": ["sqrt", "log2", 0.5],
}

param_grid_simple = {
    "rf__n_estimators": [500],
    "rf__max_depth": [None],
    "rf__min_samples_split": [5],
    "rf__min_samples_leaf": [2],
    "rf__max_features": ["sqrt"],
}


def input_example(df_train: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [col for col in df_train.columns if col not in EXCLUDE_COLS]
    x_train = df_train.drop(["time_cycles", "RUL"], axis=1)
    x_train_features = x_train[feature_cols]
    return x_train_features.iloc[[0]]


# GridSearch instead of RandomSearch to gain time - optuna later?
def fit_rf(df_train: pd.DataFrame, param_grid: dict | None = None) -> tuple[Pipeline, float | None]:
    feature_cols = [col for col in df_train.columns if col not in EXCLUDE_COLS]

    y_train = df_train["RUL"]
    x_train = df_train.drop(["time_cycles", "RUL"], axis=1)

    x_train_features = x_train[feature_cols]

    best_model: Pipeline | None = None
    if param_grid is not None:
        groups = x_train["unit_number"]  # GroupKfold - we split on engine id
        group_cv = GroupKFold(n_splits=5)
        grid_search = GridSearchCV(pipeline, param_grid, cv=group_cv, scoring=rmse_scorer, n_jobs=-1, verbose=1)
        grid_search.fit(x_train_features, y_train, groups=groups)

        print(f"Best parameters: {grid_search.best_params_}")
        rmse = -grid_search.best_score_
        print(f"Best CV RMSE: {rmse:.4f}")  # `-`: greater_is_better=False in scorer
        best_model = grid_search.best_estimator_
    else:
        best_model = pipeline.fit(x_train_features, y_train)
        rmse = None

    return best_model, rmse


# Prepare test data - only last row per engine unit for RUL prediction
def eval_rul(model: Pipeline, df_test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, Metrics]:
    # feature_cols = [col for col in df_test.columns if col not in EXCLUDE_COLS]
    # assert set(feature_cols) == set(
    #     features), f"feature_cols and features must be the same. feature_cols: {feature_cols}, features: {features}"

    x_test = df_test[features]
    y_test = df_test["RUL"]

    y_pred = model.predict(x_test)

    rmse = rmse_score(np.asarray(y_test), y_pred)
    print(f"Test RMSE: {rmse:.2f}")

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Test MAE: {mae:.2f}")
    print(f"Test R²: {r2:.3f}")

    metrics: Metrics = Metrics(rmse=rmse, r2=r2, mae=mae)

    return y_pred, np.asarray(y_test), metrics


def plot_rmse(y_test: np.ndarray, y_pred: np.ndarray, rmse: float) -> plt.Figure:
    temp_folder = config.TEMP_FOLDER
    os.makedirs(temp_folder, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    ax.set_xlabel("Actual RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title(f"Predictions vs Actual (RMSE: {rmse:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(temp_folder, "RUL_predictions_vs_actual.png"), bbox_inches="tight")

    return fig
