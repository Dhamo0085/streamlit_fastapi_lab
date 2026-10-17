"""Shared fixtures for the FastAPI lab tests.

Tests never read or write the committed model/iris_model.pkl (except the one
test that checks it on purpose). A model is trained once per session into a
temporary folder, and each test gets its own copy so it can delete or corrupt
files without affecting other tests.
"""
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import predict
import train


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory) -> Path:
    model_dir = tmp_path_factory.mktemp("trained_model")
    train.main(model_dir=model_dir)
    return model_dir


@pytest.fixture
def model_dir(trained_model_dir, tmp_path, monkeypatch) -> Path:
    """A private copy of the trained model + metrics.json that predict.py uses."""
    model_dir = tmp_path / "model"
    shutil.copytree(trained_model_dir, model_dir)
    monkeypatch.setattr(predict, "MODEL_PATH", model_dir / "iris_model.pkl")
    predict.reset_model_cache()
    yield model_dir
    predict.reset_model_cache()


@pytest.fixture
def missing_model(tmp_path, monkeypatch) -> Path:
    """Point predict.py at a model file that does not exist."""
    path = tmp_path / "missing" / "iris_model.pkl"
    monkeypatch.setattr(predict, "MODEL_PATH", path)
    predict.reset_model_cache()
    yield path
    predict.reset_model_cache()


@pytest.fixture
def client(model_dir):
    from main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_without_model(missing_model):
    from main import app

    with TestClient(app) as test_client:
        yield test_client

