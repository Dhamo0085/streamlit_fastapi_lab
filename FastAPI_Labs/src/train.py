import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from data import FEATURE_NAMES, TARGET_NAMES, load_data, split_data

# Resolve paths from this file, not the current directory, so training works
# from the repo root, from src/, or from anywhere else (the original used
# "../model/..." which only worked after `cd src`).
MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
MODEL_PATH = MODEL_DIR / "iris_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

# Original hyper-parameters, kept as the default model.
MAX_DEPTH = 3
RANDOM_STATE = 12
CV_FOLDS = 5


def build_model() -> DecisionTreeClassifier:
    """Return an untrained classifier with the original lab's settings."""
    return DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=RANDOM_STATE)


def fit_model(X_train, y_train, model_path: Path | str | None = None) -> DecisionTreeClassifier:
    """
    Train a Decision Tree Classifier and save the model to a file.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
        model_path (Path | str | None): Where to save the model. Defaults to
            FastAPI_Labs/model/iris_model.pkl.
    Returns:
        DecisionTreeClassifier: The fitted model (also written to disk).
    """
    model_path = Path(model_path) if model_path is not None else MODEL_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)

    dt_classifier = build_model()
    dt_classifier.fit(X_train, y_train)
    joblib.dump(dt_classifier, model_path)
    return dt_classifier


def evaluate_model(model: DecisionTreeClassifier, X_train, X_test, y_train, y_test) -> dict:
    """
    Compute evaluation metrics and training-data statistics for a fitted model.

    Cross-validation re-fits fresh copies of the same configuration on the
    training split only, so the held-out test set stays unseen.
    """
    y_pred = model.predict(X_test)
    cv_scores = cross_val_score(build_model(), X_train, y_train, cv=CV_FOLDS)
    X_train = np.asarray(X_train)

    return {
        "model_type": type(model).__name__,
        "params": {"max_depth": MAX_DEPTH, "random_state": RANDOM_STATE},
        "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "cv_folds": CV_FOLDS,
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist(),
        "target_names": TARGET_NAMES,
        "feature_names": FEATURE_NAMES,
        "feature_importances": {
            name: round(float(value), 4)
            for name, value in zip(FEATURE_NAMES, model.feature_importances_)
        },
        # Used by the API to warn when a request is outside what the model saw.
        "feature_ranges": {
            name: {"min": float(X_train[:, i].min()), "max": float(X_train[:, i].max())}
            for i, name in enumerate(FEATURE_NAMES)
        },
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def save_metrics(metrics: dict, metrics_path: Path | str | None = None) -> Path:
    """Write metrics as pretty-printed JSON and return the path written."""
    metrics_path = Path(metrics_path) if metrics_path is not None else METRICS_PATH
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics_path


def main(model_dir: Path | str | None = None) -> dict:
    """Train, evaluate and save the model plus metrics.json. Returns the metrics."""
    model_dir = Path(model_dir) if model_dir is not None else MODEL_DIR

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = fit_model(X_train, y_train, model_path=model_dir / MODEL_PATH.name)
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    metrics_path = save_metrics(metrics, model_dir / METRICS_PATH.name)

    print(f"Model saved to   {model_dir / MODEL_PATH.name}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Test accuracy:   {metrics['test_accuracy']:.4f} (n_test={metrics['n_test']})")
    print(
        f"{CV_FOLDS}-fold CV:       {metrics['cv_accuracy_mean']:.4f} "
        f"+/- {metrics['cv_accuracy_std']:.4f} (n_train={metrics['n_train']})"
    )
    return metrics


if __name__ == "__main__":
    main()
