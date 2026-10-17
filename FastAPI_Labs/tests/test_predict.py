"""Tests for predict.py: caching, probabilities, metrics and range warnings."""
import numpy as np
import pytest

import predict
from iris_samples import SETOSA, VERSICOLOR, VIRGINICA

ORDER = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def as_row(sample: dict) -> list[float]:
    return [sample[name] for name in ORDER]


def test_predict_data_returns_numpy_array_of_labels(model_dir):
    y_pred = predict.predict_data([as_row(SETOSA), as_row(VIRGINICA)])
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (2,)
    assert set(y_pred.tolist()) <= {0, 1, 2}


@pytest.mark.parametrize(
    "sample, expected",
    [(SETOSA, 0), (VERSICOLOR, 1), (VIRGINICA, 2)],
    ids=["setosa", "versicolor", "virginica"],
)
def test_predict_data_known_samples(model_dir, sample, expected):
    assert predict.predict_data([as_row(sample)])[0] == expected


def test_predict_proba_rows_sum_to_one(model_dir):
    proba = predict.predict_proba_data([as_row(SETOSA), as_row(VERSICOLOR), as_row(VIRGINICA)])
    assert proba.shape == (3, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_proba_argmax_matches_predict(model_dir):
    rows = [as_row(SETOSA), as_row(VERSICOLOR), as_row(VIRGINICA)]
    np.testing.assert_array_equal(
        predict.predict_proba_data(rows).argmax(axis=1), predict.predict_data(rows)
    )


def test_model_is_loaded_from_disk_only_once(model_dir, monkeypatch):
    calls = []
    real_load = predict.joblib.load
    monkeypatch.setattr(predict.joblib, "load", lambda path: calls.append(path) or real_load(path))
    predict.reset_model_cache()

    for _ in range(5):
        predict.predict_data([as_row(SETOSA)])
    predict.predict_proba_data([as_row(SETOSA)])

    assert len(calls) == 1


def test_reset_model_cache_forces_reload(model_dir, monkeypatch):
    calls = []
    real_load = predict.joblib.load
    monkeypatch.setattr(predict.joblib, "load", lambda path: calls.append(path) or real_load(path))
    predict.reset_model_cache()

    predict.load_model()
    predict.reset_model_cache()
    predict.load_model()

    assert len(calls) == 2


def test_missing_model_raises_clear_error(missing_model):
    with pytest.raises(predict.ModelNotAvailableError, match="train.py"):
        predict.predict_data([as_row(SETOSA)])


def test_model_not_available_error_is_a_file_not_found_error():
    # Callers that caught FileNotFoundError before still work.
    assert issubclass(predict.ModelNotAvailableError, FileNotFoundError)


def test_model_is_available_true_and_false(model_dir, monkeypatch, tmp_path):
    assert predict.model_is_available() is True
    monkeypatch.setattr(predict, "MODEL_PATH", tmp_path / "nope.pkl")
    assert predict.model_is_available() is False


def test_corrupt_model_is_reported_unavailable(model_dir):
    (model_dir / "iris_model.pkl").write_bytes(b"not a pickle")
    predict.reset_model_cache()
    assert predict.model_is_available() is False


def test_load_metrics(model_dir):
    metrics = predict.load_metrics()
    assert metrics["n_train"] == 105
    assert predict.metrics_path() == model_dir / "metrics.json"


def test_load_metrics_missing_or_corrupt_returns_none(model_dir):
    (model_dir / "metrics.json").write_text("{not json")
    predict.reset_model_cache()
    assert predict.load_metrics() is None
    (model_dir / "metrics.json").unlink()
    predict.reset_model_cache()
    assert predict.load_metrics() is None


def test_check_training_range_no_warnings_for_typical_input(model_dir):
    assert predict.check_training_range(SETOSA) == []


def test_check_training_range_warns_with_readable_message(model_dir):
    warnings = predict.check_training_range({**SETOSA, "sepal_length": 9.5})
    assert warnings == [
        "sepal_length=9.5 is outside the range seen in training (4.3-7.9); "
        "the prediction may be unreliable"
    ]


def test_check_training_range_without_metrics_is_silent(model_dir):
    (model_dir / "metrics.json").unlink()
    predict.reset_model_cache()
    assert predict.check_training_range({**SETOSA, "sepal_length": 9.5}) == []


def test_committed_model_loads_and_predicts(monkeypatch):
    """The model file shipped in the repo works with the pinned scikit-learn."""
    committed = predict.Path(predict.__file__).resolve().parents[1] / "model" / "iris_model.pkl"
    monkeypatch.setattr(predict, "MODEL_PATH", committed)
    predict.reset_model_cache()
    try:
        assert committed.is_file()
        assert predict.predict_data([as_row(SETOSA)])[0] == 0
        assert predict.predict_data([as_row(VIRGINICA)])[0] == 2
    finally:
        predict.reset_model_cache()
