import pandas as pd
from pydantic import BaseModel

index_names: list[str] = ["unit_number", "time_cycles"]
setting_names: list[str] = ["setting_1", "setting_2", "setting_3"]
sensor_names: list[str] = [f"s_{i + 1}" for i in range(0, 21)]


class FeatureEngineeringSettings(BaseModel):
    # model_config = SettingsConfigDict(
    #     yaml_file='FE_config.yaml',
    #     env_file_encoding='utf-8'
    # )

    rolling_windows: list[int] | None = [3, 5, 10, 20]
    delta_features: bool = True
    settings_x_settings_interaction_features: bool = True
    settings_sensor_interactions: bool = True
    temperature_group_interaction_features: bool = True
    pressure_group_interaction_features: bool = True
    speed_group_interaction_features: bool = True
    fuel_air_group_interaction_features: bool = True
    cross_system_group_interaction_features: bool = True


def create_features(df: pd.DataFrame, settings: FeatureEngineeringSettings) -> pd.DataFrame:
    df_result = df.copy()

    df_result = unique_unit_numbers(df_result)

    if settings.rolling_windows is not None:
        df_result = create_rolling_features(df_result, settings.rolling_windows)
    if settings.delta_features:
        df_result = create_delta_features(df_result)
    if settings.settings_x_settings_interaction_features:
        df_result, _ = create_settings_x_settings_interaction_features(df_result)
    if settings.settings_sensor_interactions:
        df_result = create_settings_sensor_interactions(df_result)
    if settings.temperature_group_interaction_features:
        df_result, _ = create_temperature_group_interaction_features(df_result)
    if settings.pressure_group_interaction_features:
        df_result, _ = create_pressure_group_interaction_features(df_result)
    if settings.speed_group_interaction_features:
        df_result, _ = create_speed_group_interaction_features(df_result)
    if settings.fuel_air_group_interaction_features:
        df_result, _ = create_fuel_air_group_interaction_features(df_result)
    if settings.cross_system_group_interaction_features:
        df_result, _ = create_cross_system_group_interaction_features(df_result)

    return df_result


# let's make unique unit_numbers across categories for the all dataset to avoid confusion
def unique_unit_numbers(df: pd.DataFrame) -> pd.DataFrame:
    df_result = df.copy()
    for subset_id in [1, 2, 3, 4]:
        mask = df_result["subset"] == subset_id
        df_result.loc[mask, "unit_number"] += 1000 * subset_id

    return df_result


def create_rolling_features(df: pd.DataFrame, window_sizes: list[int] = (3, 5, 10)) -> pd.DataFrame:
    df_result = df.copy()

    df_result = df_result.sort_values(["subset", "unit_number", "time_cycles"])

    sensor_cols = [col for col in df_result.columns if col.startswith("s_")]
    settings_cols = [col for col in df_result.columns if col.startswith("setting_")]

    print(f"Creating rolling features for {len(sensor_cols)} sensor and {len(settings_cols)} settings columns...")

    for window_size in window_sizes:
        print(f"  Processing window size {window_size}...")

        for col in sensor_cols + settings_cols:
            feature_name = f"{col}_roll_{window_size}"
            df_result[feature_name] = (
                df_result.groupby("unit_number")[col]
                .rolling(window=window_size, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

    return df_result


def create_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df.sort_values(["subset", "unit_number", "time_cycles"]).copy()

    feature_cols = setting_names + sensor_names
    print(f"Creating delta features for all {len(feature_cols)} columns")

    grouped = df_sorted.groupby("unit_number")
    for col in feature_cols:
        delta_name = f"{col}_delta"
        df_sorted[delta_name] = grouped[col].diff().fillna(0)

    delta_cols = len([c for c in df_sorted.columns if "_delta" in c])
    print(f"Created {delta_cols} delta features")
    return df_sorted


def create_settings_x_settings_interaction_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df_copy = df.copy()
    df_copy["setting_1_x_setting_2"] = df_copy["setting_1"] * df_copy["setting_2"]
    df_copy["setting_1_x_setting_3"] = df_copy["setting_1"] * df_copy["setting_3"]
    df_copy["setting_2_x_setting_3"] = df_copy["setting_2"] * df_copy["setting_3"]
    interaction_cols = [c for c in df_copy.columns if c not in df.columns]
    print(f"Created {len(interaction_cols)} interaction for settings features")

    return df_copy, interaction_cols


def create_settings_sensor_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()

    interaction_count = 0

    for setting in setting_names:
        for sensor in sensor_names:
            feature_name = f"{setting}_x_{sensor}"
            df_copy[feature_name] = df_copy[setting] * df_copy[sensor]
            interaction_count += 1

    print(f"Created {interaction_count} settings×sensors interaction features")
    print(f"({len(setting_names)} settings × {len(sensor_names)} sensors)")

    return df_copy


def create_temperature_group_interaction_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df_copy = df.copy()

    df_copy["temp_fan_to_lpc"] = df_copy["s_1"] * df_copy["s_2"]
    df_copy["temp_compression_ratio"] = df_copy["s_3"] / df_copy["s_2"]
    df_copy["temp_expansion_ratio"] = df_copy["s_4"] / df_copy["s_3"]
    df_copy["temp_overall_rise"] = df_copy["s_3"] - df_copy["s_1"]

    interaction_cols = [c for c in df_copy.columns if c not in df.columns]
    print(f"Created {len(interaction_cols)} interaction for temperature features")

    return df_copy, interaction_cols


def create_pressure_group_interaction_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df_copy = df.copy()

    df_copy["pressure_ratio_fan"] = df_copy["s_7"] / (df_copy["s_5"] + 1e-6)
    df_copy["pressure_bypass_vs_core"] = df_copy["s_6"] / (df_copy["s_7"] + 1e-6)
    df_copy["pressure_drop_turbine"] = df_copy["s_7"] - df_copy["s_11"]

    interaction_cols = [c for c in df_copy.columns if c not in df.columns]
    print(f"Created {len(interaction_cols)} interaction for pressure features")

    return df_copy, interaction_cols


# Speed Group Interaction Features
def create_speed_group_interaction_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df_copy = df.copy()

    df_copy["speed_fan_core_ratio"] = df_copy["s_8"] / (df_copy["s_9"] + 1e-6)
    df_copy["speed_corrected_ratio"] = df_copy["s_13"] / (df_copy["s_14"] + 1e-6)
    df_copy["speed_efficiency"] = df_copy["s_8"] / (df_copy["s_18"] + 1e-6)

    interaction_cols = [c for c in df_copy.columns if c not in df.columns]
    print(f"Created {len(interaction_cols)} interaction for speed features")

    return df_copy, interaction_cols


# Fuel & Air Group Interaction Features
def create_fuel_air_group_interaction_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df_copy = df.copy()

    df_copy["fuel_air_efficiency"] = df_copy["s_12"] * df_copy["s_16"]
    df_copy["cooling_air_total"] = df_copy["s_20"] + df_copy["s_21"]
    df_copy["fuel_to_cooling"] = df_copy["s_12"] / (df_copy["s_20"] + df_copy["s_21"] + 1e-6)

    interaction_cols = [c for c in df_copy.columns if c not in df.columns]
    print(f"Created {len(interaction_cols)} interaction for fuel & air features")

    return df_copy, interaction_cols


# Cross-System Group Interaction Features
def create_cross_system_group_interaction_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df_copy = df.copy()

    df_copy["thermal_pressure_stress"] = df_copy["s_3"] * df_copy["s_7"]
    df_copy["engine_load_indicator"] = df_copy["s_10"] * df_copy["s_15"]

    interaction_cols = [c for c in df_copy.columns if c not in df.columns]
    print(f"Created {len(interaction_cols)} interaction for cross-system features")

    return df_copy, interaction_cols


#
#
# from pydantic import BaseModel
# from pydantic_settings import BaseSettings, SettingsConfigDict
# import yaml
# from pathlib import Path
#
#
# class DatabaseSettings(BaseModel):
#     host: str
#     port: int
#     username: str
#     password: str
#
#
# class AppSettings(BaseSettings):
#     model_config = SettingsConfigDict(
#         yaml_file='config.yaml',
#         env_file_encoding='utf-8'
#     )
#
#     name: str
#     debug: bool = False
#     database: DatabaseSettings
