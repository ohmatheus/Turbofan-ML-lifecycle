from src.data_utils.load import download_kaggle_dataset, load_prepared_apply_fe, prepare_raw_data, save_prepared


def main() -> None:
    dataset_name = "behrad3d/nasa-cmaps"
    download_kaggle_dataset(dataset_name)
    datasets = prepare_raw_data()
    save_prepared(datasets)
    load_prepared_apply_fe()


if __name__ == "__main__":
    main()
