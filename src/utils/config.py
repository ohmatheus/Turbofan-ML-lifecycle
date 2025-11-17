from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_PATH = Path(__file__).parent.parent.parent

# from dotenv import load_dotenv
# load_dotenv()


class TPMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", frozen=True)

    RAW_DATA_PATH: Path = ROOT_PATH / "data/raw/"
    PROCESSED_DATA_PATH: Path = ROOT_PATH / "data/processed/"
    PREPARED_DATA_PATH: Path = ROOT_PATH / "data/prepared/"
    READY_DATA_PATH: Path = ROOT_PATH / "data/ready/"
    MODELS_PATH: Path = ROOT_PATH / "data/models/"
    FEEDBACK_PATH: Path = ROOT_PATH / "data/feedbacks/"
    TEMP_FOLDER: Path = ROOT_PATH / "_temp/"

    TEST_ENV: str = "dev"

    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    UID: int = 1000
    GID: int = 1000

    PREDICTION_POOL_PER_USER: int = 30  # random count from 1-x number of rows sent to prediction
    NUM_USERS: int = 10
    SIMULATE_ERRORS: bool = False

    DEMO_DURATION_MINUTES: int = 10 # will gradually increase from `DEMO_FIRST_TRAIN_SIZE` to 1.0 in X minutes
    DEMO_FIRST_TRAIN_SIZE: float = 0.1



config = TPMSettings()
