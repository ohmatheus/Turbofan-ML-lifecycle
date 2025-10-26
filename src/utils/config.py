from pathlib import Path

# from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_PATH = Path(__file__).parent.parent.parent

# load_dotenv()


class TPMSettings(BaseSettings):
    model_config = SettingsConfigDict(frozen=True)

    RAW_DATA_PATH: Path = ROOT_PATH / "data/raw/"
    PROCESSED_DATA_PATH: Path = ROOT_PATH / "data/processed/"
    PREPARED_DATA_PATH: Path = ROOT_PATH / "data/prepared/"

    MLFLOW_TRACKING_URI: str = "http://localhost:5000"  # need to run `mlflow server` to start server


config = TPMSettings()
