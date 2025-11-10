import logging
import os
from pathlib import Path

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore[import-untyped]

from src.data_utils.feature_engineering import FeatureEngineeringSettings, create_features
from src.utils.config import config

logger = logging.getLogger(__name__)

subsets = ["001", "002", "003", "004"]


def download_kaggle_dataset(dataset_name: str, download_path: Path = config.RAW_DATA_PATH) -> None:
    os.makedirs(download_path, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    # Download the dataset
    try:
        logger.info(f"Downloading {dataset_name}...")
        api.dataset_download_files(dataset=dataset_name, path=download_path, unzip=True, quiet=False)
        logger.info(f"Successfully downloaded to {download_path}")

        # List downloaded files
        for root, _dirs, files in os.walk(download_path):
            for file in files:
                logger.info(f"Downloaded: {os.path.join(root, file)}")

    except Exception as e:
        logger.exception(f"Error downloading dataset: {e}")


def _add_rul_column(df: pd.DataFrame, ref_rul: pd.DataFrame | None = None) -> pd.DataFrame:
    train_grouped_by_unit = df.groupby(by="unit_number")
    max_time_cycles = train_grouped_by_unit["time_cycles"].max()
    merged = df.merge(max_time_cycles.to_frame(name="max_time_cycle"), left_on="unit_number", right_index=True)

    if ref_rul is not None:
        # For test data with reference RUL
        # Extract values from DataFrame (first column)
        ref_rul_values = ref_rul.iloc[:, 0]

        unique_units = df["unit_number"].nunique()
        assert len(ref_rul_values) == unique_units, f"RUL count ({len(ref_rul_values)}) ≠ unique units ({unique_units})"

        unit_numbers = sorted(df["unit_number"].unique())
        rul_mapping = dict(zip(unit_numbers, ref_rul_values, strict=True))
        merged["ref_rul"] = merged["unit_number"].map(rul_mapping)
        merged["RUL"] = merged["ref_rul"] + (merged["max_time_cycle"] - merged["time_cycles"])
        merged = merged.drop(["max_time_cycle", "ref_rul"], axis=1)
    else:
        # For training data
        merged["RUL"] = merged["max_time_cycle"] - merged["time_cycles"]
        merged = merged.drop("max_time_cycle", axis=1)

    return merged


def prepare_raw_data() -> dict[str, dict[str, pd.DataFrame]]:
    index_names = ["unit_number", "time_cycles"]
    setting_names = ["setting_1", "setting_2", "setting_3"]
    sensor_names = [f"s_{i + 1}" for i in range(0, 21)]
    col_names = index_names + setting_names + sensor_names

    raw_data_path = config.RAW_DATA_PATH / "CMaps/"

    datasets = {}

    for fd_num in subsets:
        datasets[fd_num] = {
            "train": pd.read_csv(
                raw_data_path / f"train_FD{fd_num}.txt", sep=r"\s+", header=None, index_col=False, names=col_names
            ),
            "test": pd.read_csv(
                raw_data_path / f"test_FD{fd_num}.txt", sep=r"\s+", header=None, index_col=False, names=col_names
            ),
            "rul": pd.read_csv(
                raw_data_path / f"RUL_FD{fd_num}.txt", sep=r"\s+", header=None, index_col=False, names=["RUL"]
            ),
        }

    # Apply to all datasets
    for fd_num in subsets:
        datasets[fd_num]["train"] = _add_rul_column(datasets[fd_num]["train"])
        datasets[fd_num]["test"] = _add_rul_column(datasets[fd_num]["test"], datasets[fd_num]["rul"])

    return datasets


def save_prepared(datasets: dict[str, dict[str, pd.DataFrame]]) -> None:
    train_data = []
    test_data = []

    for fd_num in subsets:
        train_subset = datasets[fd_num]["train"].copy()
        train_subset["subset"] = fd_num
        train_data.append(train_subset)

        test_subset = datasets[fd_num]["test"].copy()
        test_subset["subset"] = fd_num
        test_data.append(test_subset)

    # Concatenate all train and test dataframes
    train_df = pd.concat(train_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)

    processed_dir = config.PREPARED_DATA_PATH
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(processed_dir / "train-all-prepared.csv", index=False)
    test_df.to_csv(processed_dir / "test-all-prepared.csv", index=False)

    logger.info(f"Train dataset shape: {train_df.shape}")
    logger.info(f"Test dataset shape: {test_df.shape}")
    logger.info("CSV files saved successfully.")


def load_prepared_apply_fe() -> None:
    fe_settings = FeatureEngineeringSettings()

    prepared_folder = config.PREPARED_DATA_PATH
    train_df = pd.read_csv(prepared_folder / "train-all-prepared.csv", index_col=False)
    test_df = pd.read_csv(prepared_folder / "test-all-prepared.csv", index_col=False)

    train_df = create_features(train_df, fe_settings)
    test_df = create_features(test_df, fe_settings)

    rul_thresholds = {
        1: {"max": 145, "min": 6},
        2: {"max": 194, "min": 6},
        3: {"max": 145, "min": 6},
        4: {"max": 194, "min": 6},
    }

    # Apply different RUL filtering for each subset
    filtered_dfs = []
    for subset_id in [1, 2, 3, 4]:
        subset_data = train_df[train_df["subset"] == subset_id]
        max_rul = rul_thresholds[subset_id]["max"]
        min_rul = rul_thresholds[subset_id]["min"]

        filtered_subset = subset_data[(subset_data["RUL"] <= max_rul) & (subset_data["RUL"] >= min_rul)]
        filtered_dfs.append(filtered_subset)

    # Combine all filtered subsets back together
    train_df = pd.concat(filtered_dfs, ignore_index=True)

    config.READY_DATA_PATH.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(config.READY_DATA_PATH / "train.csv", index=False)
    test_df.to_csv(config.READY_DATA_PATH / "test.csv", index=False)
    logger.info("Train and Test dataframes are ready to use for simulation.")
