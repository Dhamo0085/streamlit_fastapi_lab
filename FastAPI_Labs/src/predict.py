import json
from functools import lru_cache
from pathlib import Path

import joblib

# Absolute path so the API works no matter which directory uvicorn is started
# from. Tests monkeypatch this module attribute to point at a temporary model.
MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "iris_model.pkl"


class ModelNotAvailableError(FileNotFoundError):
    """Raised when the trained model file does not exist yet."""


@lru_cache(maxsize=4)
def _load_model_from(path: str):
    # Cached per path: the model is read from disk once, not on every request.
    return joblib.load(path)


def load_model():
    """
    Return the trained model, loading it from MODEL_PATH on first use.
    Raises:
        ModelNotAvailableError: If the model file is missing (run train.py).
    """
    path = Path(MODEL_PATH)
    if not path.is_file():
        raise ModelNotAvailableError(
            f"Model file not found at {path}. Run `python FastAPI_Labs/src/train.py` first."
        )
    return _load_model_from(str(path))


def reset_model_cache() -> None:
    """Forget any cached model/metrics so the next call re-reads them from disk."""
    _load_model_from.cache_clear()
    _load_metrics_from.cache_clear()


def metrics_path() -> Path:
    """metrics.json lives next to the model file written by train.py."""
    return Path(MODEL_PATH).with_name("metrics.json")


@lru_cache(maxsize=4)
def _load_metrics_from(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_metrics() -> dict | None:
    """Return the training metrics, or None if metrics.json is missing or unreadable."""
    path = metrics_path()
    if not path.is_file():
        return None
    try:
        return _load_metrics_from(str(path))
    except (OSError, ValueError):
        return None


def check_training_range(sample: dict[str, float]) -> list[str]:
    """
    Return a warning for each feature outside the range seen during training.

    These inputs are still valid (they passed the API's hard limits); the
    warning only tells the caller that the model is extrapolating.
    """
    metrics = load_metrics() or {}
    ranges = metrics.get("feature_ranges", {})
    warnings = []
    for name, value in sample.items():
        bounds = ranges.get(name)
        if bounds and not (bounds["min"] <= value <= bounds["max"]):
            warnings.append(
                f"{name}={float(value)} is outside the range seen in training "
                f"({float(bounds['min'])}-{float(bounds['max'])}); the prediction may be unreliable"
            )
    return warnings


def model_is_available() -> bool:
    """Return True if the model can be loaded, False otherwise (never raises)."""
    try:
        load_model()
    except Exception:
        return False
    return True


def predict_data(X):
    """
    Predict the class labels for the input data.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    model = load_model()
    y_pred = model.predict(X)
    return y_pred


def predict_proba_data(X):
    """
    Predict class probabilities for the input data.
    Args:
        X (numpy.ndarray): Input data, shape (n_samples, 4).
    Returns:
        numpy.ndarray: Shape (n_samples, 3); each row sums to 1.
    """
    model = load_model()
    return model.predict_proba(X)
