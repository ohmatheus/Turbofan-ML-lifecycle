from src.data.load import download_kaggle_dataset, prepare_raw_data, save_prepared


def main() -> None:
    dataset_name = "behrad3d/nasa-cmaps"
    download_kaggle_dataset(dataset_name)
    datasets = prepare_raw_data()
    save_prepared(datasets)


if __name__ == "__main__":
    main()
