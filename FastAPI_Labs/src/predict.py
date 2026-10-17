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
    """Forget any cached model so the next call re-reads it from disk."""
    _load_model_from.cache_clear()


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
