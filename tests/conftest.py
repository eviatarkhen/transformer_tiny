import pytest

from tiny.config import Config, model_dir
from tiny.weights import load_weights


@pytest.fixture(scope="session")
def config() -> Config:
    return Config.from_json(model_dir() / "config.json")


@pytest.fixture(scope="session")
def weights() -> dict:
    return load_weights()
