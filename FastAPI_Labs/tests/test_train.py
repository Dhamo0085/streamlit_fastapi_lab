"""Tests for train.py: model settings, saved artefacts and metrics."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import pytest
from sklearn.tree import DecisionTreeClassifier

import train
from data import FEATURE_NAMES, TARGET_NAMES, load_data, split_data

SRC_DIR = Path(train.__file__).resolve().parent

EXPECTED_METRIC_KEYS = {
    "model_type",
    "params",
    "test_accuracy",
    "cv_folds",
    "cv_accuracy_mean",
    "cv_accuracy_std",
    "confusion_matrix",
    "target_names",
    "feature_names",
    "feature_importances",
    "feature_ranges",
    "n_train",
    "n_test",
    "sklearn_version",
    "trained_at",
}


@pytest.fixture(scope="module")
def metrics(trained_model_dir):
    return json.loads((trained_model_dir / "metrics.json").read_text())


def test_main_creates_model_and_metrics(trained_model_dir):
    assert (trained_model_dir / "iris_model.pkl").is_file()
    assert (trained_model_dir / "metrics.json").is_file()


def test_metrics_has_all_keys(metrics):
    assert EXPECTED_METRIC_KEYS <= metrics.keys()


def test_metrics_values_are_sane(metrics):
    assert 0.85 <= metrics["test_accuracy"] <= 1.0
    assert 0.85 <= metrics["cv_accuracy_mean"] <= 1.0
    assert metrics["cv_accuracy_std"] >= 0
    assert metrics["n_train"] == 105
    assert metrics["n_test"] == 45
    assert metrics["target_names"] == TARGET_NAMES
    assert metrics["feature_names"] == FEATURE_NAMES


def test_confusion_matrix_covers_whole_test_set(metrics):
    matrix = metrics["confusion_matrix"]
    assert len(matrix) == 3 and all(len(row) == 3 for row in matrix)
    assert sum(map(sum, matrix)) == metrics["n_test"]


def test_feature_importances_sum_to_one(metrics):
    assert set(metrics["feature_importances"]) == set(FEATURE_NAMES)
    assert sum(metrics["feature_importances"].values()) == pytest.approx(1.0, abs=1e-3)


def test_feature_ranges_match_training_split(metrics):
    X_train, _, _, _ = split_data(*load_data())
    for i, name in enumerate(FEATURE_NAMES):
        assert metrics["feature_ranges"][name] == {
            "min": pytest.approx(X_train[:, i].min()),
            "max": pytest.approx(X_train[:, i].max()),
        }


def test_fit_model_keeps_original_hyperparameters(tmp_path):
    X_train, _, y_train, _ = split_data(*load_data())
    model = train.fit_model(X_train, y_train, model_path=tmp_path / "m.pkl")
    assert isinstance(model, DecisionTreeClassifier)
    assert model.max_depth == 3
    assert model.random_state == 12


def test_fit_model_saved_file_matches_returned_model(tmp_path):
    X_train, X_test, y_train, _ = split_data(*load_data())
    model = train.fit_model(X_train, y_train, model_path=tmp_path / "nested" / "m.pkl")
    loaded = joblib.load(tmp_path / "nested" / "m.pkl")
    assert (loaded.predict(X_test) == model.predict(X_test)).all()


def test_default_paths_are_absolute_and_inside_the_lab():
    assert train.MODEL_PATH.is_absolute()
    assert train.MODEL_PATH == SRC_DIR.parent / "model" / "iris_model.pkl"
    assert train.METRICS_PATH == SRC_DIR.parent / "model" / "metrics.json"


def test_training_is_deterministic(tmp_path):
    first = train.main(model_dir=tmp_path / "a")
    second = train.main(model_dir=tmp_path / "b")
    for key in ("test_accuracy", "cv_accuracy_mean", "confusion_matrix", "feature_importances"):
        assert first[key] == second[key]


def test_train_script_runs_from_a_different_working_directory(tmp_path):
    # Copy the scripts into a throwaway lab folder so the run cannot touch the
    # committed model, then launch it from an unrelated directory (bug B2).
    lab = tmp_path / "lab"
    (lab / "src").mkdir(parents=True)
    for name in ("data.py", "train.py"):
        shutil.copy(SRC_DIR / name, lab / "src" / name)
    elsewhere = tmp_path / "somewhere_else"
    elsewhere.mkdir()

    result = subprocess.run(
        [sys.executable, str(lab / "src" / "train.py")],
        cwd=elsewhere,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert (lab / "model" / "iris_model.pkl").is_file()
    assert (lab / "model" / "metrics.json").is_file()
    assert "Test accuracy" in result.stdout
