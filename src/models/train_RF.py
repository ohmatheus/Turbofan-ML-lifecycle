import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error

from src.utils.config import config


EXCLUDE_COLS = ["subset", "unit_number", "time_cycles", "RUL"]

pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('rf', RandomForestRegressor(random_state=46, n_jobs=-1))
])

def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse_score, greater_is_better=False)

param_grid = {
    'rf__n_estimators': [300, 500, 700],
    'rf__max_depth': [None, 10, 20, 30],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4],
    'rf__max_features': ['sqrt', 'log2', 0.5]
}

param_grid_simple = {
    'rf__n_estimators': [500],
    'rf__max_depth': [None],
    'rf__min_samples_split': [5],
    'rf__min_samples_leaf': [2],
    'rf__max_features': ['sqrt']
}

#GridSearch instead of RandomSearch to gain time - optuna later?
def fit_rf(df_train: pd.DataFrame, param_grid: dict | None = None) -> tuple[Pipeline, float | None]:
    feature_cols = [col for col in df_train.columns if col not in EXCLUDE_COLS]

    y_train = df_train["RUL"]
    X_train = df_train.drop(["time_cycles", "RUL"], axis=1)

    X_train_features = X_train[feature_cols]

    best_model: Pipeline | None = None
    if param_grid is not None:
        groups = X_train['unit_number'] #GroupKfold - we split on engine id
        group_cv = GroupKFold(n_splits=5)
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=group_cv,
            scoring=rmse_scorer,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train_features, y_train, groups=groups)

        print(f"Best parameters: {grid_search.best_params_}")
        rmse = -grid_search.best_score_
        print(f"Best CV RMSE: {rmse:.4f}") # `-`: greater_is_better=False in scorer
        best_model = grid_search.best_estimator_
    else:
        best_model = pipeline.fit(X_train_features, y_train)
        rmse = None

    return best_model, rmse


# Prepare test data - only last row per engine unit for RUL prediction
def eval_rul(model, df_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    feature_cols = [col for col in df_test.columns if col not in EXCLUDE_COLS]

    # Assert that for each unit, there's only one row with max time_cycles and min RUL
    for unit in df_test["unit_number"].unique():
        unit_data = df_test[df_test["unit_number"] == unit]
        max_time_rows = unit_data[unit_data["time_cycles"] == unit_data["time_cycles"].max()]
        min_rul_rows = unit_data[unit_data["RUL"] == unit_data["RUL"].min()]

        assert len(max_time_rows) == 1, f"Unit {unit}: Multiple rows with same max time_cycles"
        assert len(min_rul_rows) == 1, f"Unit {unit}: Multiple rows with same min RUL"

        # Verify that max time_cycles and min RUL are in the same row
        assert max_time_rows.index.equals(min_rul_rows.index), f"Unit {unit}: Max time_cycles and min RUL are not in the same row"

    # Get only the last row (highest time_cycles) for each engine unit
    test_last_rows = df_test.loc[df_test.groupby("unit_number")["time_cycles"].idxmax()]

    print(f"Original test data shape: {df_test.shape}")
    print(f"Test data (last rows only) shape: {test_last_rows.shape}")

    X_test = test_last_rows[feature_cols]
    y_test = test_last_rows["RUL"]

    y_pred = model.predict(X_test)

    rmse = rmse_score(y_test, y_pred)
    print(f"Test RMSE: {rmse:.2f}")

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Test MAE: {mae:.2f}")
    print(f"Test R²: {r2:.3f}")

    return y_pred, y_test.values, rmse


def plot_rmse(y_test, y_pred, rmse):
    temp_folder = config.TEMP_FOLDER
    os.makedirs(temp_folder, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    ax.set_xlabel("Actual RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title(f"Predictions vs Actual (RMSE: {rmse:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(temp_folder, 'RUL_predictions_vs_actual.png'), dpi=300, bbox_inches='tight')

    return fig