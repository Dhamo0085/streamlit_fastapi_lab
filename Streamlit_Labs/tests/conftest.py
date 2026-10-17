"""Shared fixtures for the Streamlit lab tests.

`fake_backend` replaces requests.get/requests.post with an in-memory router,
so the client helpers and the dashboard can be tested without a server.
"""
import pytest
import requests

from dashboard_fakes import MODEL_INFO, FakeBackend, FakeResponse


@pytest.fixture(autouse=True)
def default_backend_url(monkeypatch):
    # Tests must not depend on the developer's shell environment.
    monkeypatch.delenv("FASTAPI_BACKEND_URL", raising=False)


@pytest.fixture
def fake_backend(monkeypatch) -> FakeBackend:
    backend = FakeBackend()
    monkeypatch.setattr(requests, "get", backend.get)
    monkeypatch.setattr(requests, "post", backend.post)
    return backend


@pytest.fixture
def online_backend(fake_backend) -> FakeBackend:
    """A healthy backend that predicts setosa with full confidence."""
    fake_backend.set("GET /", FakeResponse(200, {"status": "healthy", "model_loaded": True}))
    fake_backend.set("GET /model-info", FakeResponse(200, MODEL_INFO))
    fake_backend.set(
        "POST /predict/proba",
        lambda kw: FakeResponse(200, {
            "response": 0,
            "species": "setosa",
            "probabilities": {"setosa": 1.0, "versicolor": 0.0, "virginica": 0.0},
            "warnings": [],
        }),
    )
    fake_backend.set("POST /predict", FakeResponse(200, {"response": 0, "species": "setosa", "warnings": []}))
    return fake_backend


@pytest.fixture
def offline_backend(fake_backend) -> FakeBackend:
    for route in ("GET /", "GET /model-info", "POST /predict", "POST /predict/proba", "POST /predict/batch"):
        fake_backend.set(route, requests.ConnectionError("connection refused"))
    return fake_backend
