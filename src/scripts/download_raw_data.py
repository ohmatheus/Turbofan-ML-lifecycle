import os
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore[import-untyped]

from src.utils.config import config


def download_kaggle_dataset(dataset_name: str, download_path: Path = config.RAW_DATA_PATH) -> None:
    os.makedirs(download_path, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    # Download the dataset
    try:
        print(f"Downloading {dataset_name}...")
        api.dataset_download_files(dataset=dataset_name, path=download_path, unzip=True, quiet=False)
        print(f"Successfully downloaded to {download_path}")

        # List downloaded files
        for root, _dirs, files in os.walk(download_path):
            for file in files:
                print(f"Downloaded: {os.path.join(root, file)}")

    except Exception as e:
        print(f"Error downloading dataset: {e}")


def main() -> None:
    dataset_name = "behrad3d/nasa-cmaps"
    download_kaggle_dataset(dataset_name)


if __name__ == "__main__":
    main()
