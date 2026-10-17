"""Test doubles and sample data shared by the Streamlit lab tests."""
from pathlib import Path
from urllib.parse import urlsplit

LAB_DIR = Path(__file__).resolve().parents[1]
DASHBOARD = LAB_DIR / "src" / "Dashboard.py"
DATA_DIR = LAB_DIR / "data"

SETOSA = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}

MODEL_INFO = {
    "model_type": "DecisionTreeClassifier",
    "max_depth": 3,
    "classes": ["setosa", "versicolor", "virginica"],
    "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
    "metrics_available": True,
    "metrics": {
        "test_accuracy": 0.9556,
        "cv_folds": 5,
        "cv_accuracy_mean": 0.9429,
        "cv_accuracy_std": 0.0356,
        "confusion_matrix": [[20, 0, 0], [0, 12, 2], [0, 0, 11]],
        "target_names": ["setosa", "versicolor", "virginica"],
        "feature_importances": {"sepal_length": 0.0, "sepal_width": 0.0, "petal_length": 0.4553, "petal_width": 0.5447},
        "feature_ranges": {
            "sepal_length": {"min": 4.3, "max": 7.9},
            "sepal_width": {"min": 2.0, "max": 4.1},
            "petal_length": {"min": 1.0, "max": 6.9},
            "petal_width": {"min": 0.1, "max": 2.5},
        },
        "n_train": 105,
        "n_test": 45,
        "sklearn_version": "1.5.1",
        "trained_at": "2026-01-01T00:00:00+00:00",
    },
}


class FakeResponse:
    def __init__(self, status_code: int, body=None, text_body: str | None = None):
        self.status_code = status_code
        self._body = body
        self._text = text_body

    def json(self):
        if self._text is not None:
            raise ValueError("not JSON")
        return self._body


class FakeBackend:
    """Routes 'METHOD /path' to a FakeResponse, or raises a configured exception."""

    def __init__(self):
        self.calls: list[dict] = []
        self.routes: dict[str, object] = {}

    def set(self, route: str, response_or_exc) -> None:
        self.routes[route] = response_or_exc

    def _handle(self, method: str, url: str, **kwargs):
        path = urlsplit(url).path or "/"
        self.calls.append({"method": method, "url": url, "path": path, **kwargs})
        handler = self.routes.get(f"{method} {path}")
        if handler is None:
            return FakeResponse(404, {"detail": "Not Found"})
        if isinstance(handler, Exception):
            raise handler
        if callable(handler):
            return handler(kwargs)
        return handler

    def get(self, url, **kwargs):
        return self._handle("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self._handle("POST", url, **kwargs)
