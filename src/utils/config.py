from pathlib import Path

# from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_PATH = Path(__file__).parent.parent.parent

# load_dotenv()


class TPMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", frozen=True)

    RAW_DATA_PATH: Path = ROOT_PATH / "data/raw/"
    PROCESSED_DATA_PATH: Path = ROOT_PATH / "data/processed/"
    PREPARED_DATA_PATH: Path = ROOT_PATH / "data/prepared/"
    READY_DATA_PATH: Path = ROOT_PATH / "data/ready/"
    MODELS_PATH: Path = ROOT_PATH / "data/models/"
    TEMP_FOLDER: Path = ROOT_PATH / "_temp/"

    TEST_ENV: str = "dev"

    MLFLOW_TRACKING_URI: str = "http://localhost:5000"  # need to run `mlflow server/ui` to start server


config = TPMSettings()
