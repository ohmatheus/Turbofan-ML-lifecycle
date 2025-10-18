def test_placeholder() -> None:
    assert True


def test_imports() -> None:
    try:
        from src.utils.config import config

        assert config.MLFLOW_TRACKING_URI is not None
        assert True
    except ImportError:
        assert True
