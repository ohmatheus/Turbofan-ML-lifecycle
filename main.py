from src.utils.config import config


def print_hi(name: str) -> None:
    print(f"Hi, {name}")
    print(f"{config.MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    print_hi("hello worldasdasdasd")
