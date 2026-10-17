"""HTTP-level tests for the FastAPI app (main.py) using TestClient."""
import logging
import math

import pytest
from fastapi import HTTPException

import main
import predict
from iris_samples import SETOSA, VERSICOLOR, VIRGINICA

FIELDS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
JSON_HEADERS = {"Content-Type": "application/json"}


# --------------------------------------------------------------------------- health

def test_health_returns_healthy(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy", "model_loaded": True}


def test_health_reports_missing_model_without_failing(client_without_model):
    r = client_without_model.get("/")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy", "model_loaded": False}


def test_startup_logs_warning_when_model_missing(missing_model, caplog):
    from fastapi.testclient import TestClient

    with caplog.at_level(logging.WARNING, logger="iris_api"):
        with TestClient(main.app):
            pass
    assert any("Model not available at startup" in rec.message for rec in caplog.records)


def test_app_metadata_and_swagger_example(client):
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == "Iris Classifier API"
    assert schema["info"]["version"] == "2.0.0"
    assert schema["components"]["schemas"]["IrisData"]["examples"][0] == SETOSA


# --------------------------------------------------------------- backward compatibility

def test_original_request_body_still_works(client):
    # Exact body from the original lab. The original server answered
    # {"response": 0}; that field must keep its name, type and value.
    r = client.post("/predict", json=SETOSA)
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["response"], int)
    assert body["response"] == 0


def test_original_field_order_and_numeric_strings_still_accepted(client):
    # The original README documents that "5.1" (a string) is converted to 5.1.
    raw = '{"petal_length": "1.4", "sepal_length": "5.1", "petal_width": 0.2, "sepal_width": 3.5}'
    r = client.post("/predict", content=raw, headers=JSON_HEADERS)
    assert r.status_code == 200
    assert r.json()["response"] == 0


def test_integer_values_are_accepted(client):
    r = client.post("/predict", json={"sepal_length": 6, "sepal_width": 3, "petal_length": 5, "petal_width": 2})
    assert r.status_code == 200


# ---------------------------------------------------------------------- /predict

@pytest.mark.parametrize(
    "sample, class_id, species",
    [(SETOSA, 0, "setosa"), (VERSICOLOR, 1, "versicolor"), (VIRGINICA, 2, "virginica")],
    ids=["setosa", "versicolor", "virginica"],
)
def test_predict_known_species(client, sample, class_id, species):
    r = client.post("/predict", json=sample)
    assert r.status_code == 200
    assert r.json() == {"response": class_id, "species": species, "warnings": []}


# ------------------------------------------------------------------- validation (422)

def _with(**changes):
    return {**SETOSA, **changes}


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({k: v for k, v in SETOSA.items() if k != field}, id=f"missing-{field}")
        for field in FIELDS
    ]
    + [
        pytest.param(_with(sepal_length=-1.0), id="negative"),
        pytest.param(_with(petal_width=0), id="zero"),
        pytest.param(_with(petal_length=10.01), id="above-10"),
        pytest.param(_with(sepal_width=1000), id="huge"),
        pytest.param(_with(sepal_length="abc"), id="non-numeric-string"),
        pytest.param(_with(sepal_length=None), id="null"),
        pytest.param(_with(sepal_length=[5.1]), id="list"),
        pytest.param(_with(sepal_length={"v": 5.1}), id="object"),
        pytest.param(_with(sepal_length=True), id="boolean"),
        pytest.param(_with(color="purple"), id="extra-field"),
        pytest.param({}, id="empty-object"),
        pytest.param([SETOSA], id="array-instead-of-object"),
    ],
)
def test_predict_rejects_invalid_input(client, body):
    r = client.post("/predict", json=body)
    assert r.status_code == 422
    assert "detail" in r.json()


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_predict_rejects_non_finite_numbers_with_valid_json_error(client, token):
    # Python's json accepts NaN/Infinity; they must be a clean 422, not a 500.
    raw = '{"sepal_length": %s, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}' % token
    r = client.post("/predict", content=raw, headers=JSON_HEADERS)
    assert r.status_code == 422
    error = r.json()["detail"][0]
    assert error["loc"] == ["body", "sepal_length"]
    assert error["type"] == "finite_number"


def test_predict_rejects_empty_body(client):
    r = client.post("/predict")
    assert r.status_code == 422


def test_predict_rejects_non_json_body(client):
    r = client.post("/predict", content="sepal_length=5.1", headers=JSON_HEADERS)
    assert r.status_code == 422


def test_validation_error_names_the_bad_field(client):
    r = client.post("/predict", json=_with(petal_width=-3))
    error = r.json()["detail"][0]
    assert error["loc"] == ["body", "petal_width"]
    assert error["type"] == "greater_than"


def test_boolean_error_message_is_readable(client):
    r = client.post("/predict", json=_with(sepal_length=False))
    assert r.json()["detail"][0]["ctx"]["error"] == "must be a number, not a boolean"


@pytest.mark.parametrize("value", [0.001, 0.01, 9.99, 10.0])
@pytest.mark.parametrize("field", FIELDS)
def test_boundary_values_just_inside_limits_are_accepted(client, field, value):
    r = client.post("/predict", json=_with(**{field: value}))
    assert r.status_code == 200


# ----------------------------------------------------------------- soft warnings

def test_no_warnings_for_typical_input(client):
    assert client.post("/predict", json=VERSICOLOR).json()["warnings"] == []


@pytest.mark.parametrize(
    "field, value",
    [("sepal_length", 9.5), ("sepal_width", 4.4), ("petal_length", 0.5), ("petal_width", 3.0)],
)
def test_out_of_training_range_input_is_accepted_with_warning(client, field, value):
    r = client.post("/predict", json=_with(**{field: value}))
    assert r.status_code == 200
    warnings = r.json()["warnings"]
    assert len(warnings) == 1
    assert warnings[0].startswith(f"{field}={float(value)} is outside the range seen in training")


def test_warning_text_matches_contract(client):
    r = client.post("/predict", json=_with(sepal_length=9.5))
    assert r.json()["warnings"] == [
        "sepal_length=9.5 is outside the range seen in training (4.3-7.9); "
        "the prediction may be unreliable"
    ]


def test_multiple_warnings_are_all_reported(client):
    body = {"sepal_length": 9.0, "sepal_width": 5.0, "petal_length": 8.0, "petal_width": 3.0}
    assert len(client.post("/predict", json=body).json()["warnings"]) == 4


def test_no_warnings_when_metrics_file_missing(client, model_dir):
    (model_dir / "metrics.json").unlink()
    predict.reset_model_cache()
    r = client.post("/predict", json=_with(sepal_length=9.5))
    assert r.status_code == 200
    assert r.json()["warnings"] == []


# ------------------------------------------------------------- 503 / 500 handling

@pytest.mark.parametrize(
    "method, path, body",
    [
        ("post", "/predict", SETOSA),
        ("post", "/predict/proba", SETOSA),
        ("post", "/predict/batch", {"samples": [SETOSA]}),
        ("get", "/model-info", None),
    ],
)
def test_endpoints_return_503_when_model_missing(client_without_model, method, path, body):
    r = getattr(client_without_model, method)(path, **({"json": body} if body else {}))
    assert r.status_code == 503
    assert "train.py" in r.json()["detail"]


def test_validation_still_runs_before_model_check(client_without_model):
    # Bad input is a client error even when the model is missing.
    assert client_without_model.post("/predict", json=_with(sepal_length=-1)).status_code == 422


def test_unexpected_error_returns_500(client, monkeypatch):
    def boom(_):
        raise RuntimeError("something broke")

    monkeypatch.setattr(main, "predict_data", boom)
    r = client.post("/predict", json=SETOSA)
    assert r.status_code == 500
    assert r.json()["detail"] == "something broke"


def test_http_exceptions_are_not_swallowed_into_500(client, monkeypatch):
    def teapot(_):
        raise HTTPException(status_code=418, detail="teapot")

    monkeypatch.setattr(main, "predict_data", teapot)
    r = client.post("/predict", json=SETOSA)
    assert r.status_code == 418


# --------------------------------------------------------------------- /predict/proba

def test_proba_response_shape(client):
    body = client.post("/predict/proba", json=SETOSA).json()
    assert set(body) == {"response", "species", "probabilities", "warnings"}
    assert set(body["probabilities"]) == {"setosa", "versicolor", "virginica"}


@pytest.mark.parametrize("sample", [SETOSA, VERSICOLOR, VIRGINICA], ids=["setosa", "versicolor", "virginica"])
def test_proba_sums_to_one_and_argmax_matches_predict(client, sample):
    proba = client.post("/predict/proba", json=sample).json()
    single = client.post("/predict", json=sample).json()
    assert math.isclose(sum(proba["probabilities"].values()), 1.0, abs_tol=1e-9)
    assert max(proba["probabilities"], key=proba["probabilities"].get) == single["species"]
    assert proba["response"] == single["response"]


def test_proba_applies_same_validation(client):
    assert client.post("/predict/proba", json=_with(sepal_length=0)).status_code == 422


def test_proba_includes_warnings(client):
    assert client.post("/predict/proba", json=_with(sepal_length=9.5)).json()["warnings"]


# --------------------------------------------------------------------- /predict/batch

def test_batch_normal(client):
    r = client.post("/predict/batch", json={"samples": [SETOSA, VERSICOLOR, VIRGINICA]})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    assert [p["index"] for p in body["predictions"]] == [0, 1, 2]
    assert [p["species"] for p in body["predictions"]] == ["setosa", "versicolor", "virginica"]
    assert [p["response"] for p in body["predictions"]] == [0, 1, 2]


def test_batch_single_item(client):
    r = client.post("/predict/batch", json={"samples": [SETOSA]})
    assert r.status_code == 200
    assert r.json() == {
        "count": 1,
        "predictions": [{"index": 0, "response": 0, "species": "setosa", "warnings": []}],
    }


def test_batch_of_100_is_accepted(client):
    r = client.post("/predict/batch", json={"samples": [SETOSA] * 100})
    assert r.status_code == 200
    assert r.json()["count"] == 100


def test_batch_of_101_is_rejected(client):
    assert client.post("/predict/batch", json={"samples": [SETOSA] * 101}).status_code == 422


@pytest.mark.parametrize(
    "body",
    [{"samples": []}, {}, {"samples": SETOSA}, {"samples": [SETOSA], "extra": 1}],
    ids=["empty-list", "missing-samples", "not-a-list", "extra-field"],
)
def test_batch_rejects_bad_envelope(client, body):
    assert client.post("/predict/batch", json=body).status_code == 422


def test_batch_with_one_invalid_item_is_rejected_and_points_at_it(client):
    r = client.post("/predict/batch", json={"samples": [SETOSA, _with(petal_width=-1), VIRGINICA]})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", "samples", 1, "petal_width"]


def test_batch_warnings_are_per_sample(client):
    r = client.post("/predict/batch", json={"samples": [SETOSA, _with(sepal_length=9.5)]})
    predictions = r.json()["predictions"]
    assert predictions[0]["warnings"] == []
    assert len(predictions[1]["warnings"]) == 1


def test_batch_matches_single_predictions(client):
    samples = [SETOSA, VERSICOLOR, VIRGINICA, _with(petal_length=4.9, petal_width=1.7)]
    batch = client.post("/predict/batch", json={"samples": samples}).json()["predictions"]
    singles = [client.post("/predict", json=s).json()["response"] for s in samples]
    assert [p["response"] for p in batch] == singles


# ----------------------------------------------------------------------- /model-info

def test_model_info_with_metrics(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_type"] == "DecisionTreeClassifier"
    assert body["max_depth"] == 3
    assert body["classes"] == ["setosa", "versicolor", "virginica"]
    assert body["feature_names"] == FIELDS
    assert body["metrics_available"] is True
    for key in ("test_accuracy", "cv_accuracy_mean", "cv_accuracy_std", "confusion_matrix",
                "feature_importances", "feature_ranges", "trained_at", "sklearn_version"):
        assert key in body["metrics"]


def test_model_info_without_metrics_file(client, model_dir):
    (model_dir / "metrics.json").unlink()
    predict.reset_model_cache()
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["metrics_available"] is False
    assert body["metrics"] is None
    assert body["model_type"] == "DecisionTreeClassifier"


# ---------------------------------------------------------------------- routing

def test_unknown_route_returns_404(client):
    assert client.get("/does-not-exist").status_code == 404


@pytest.mark.parametrize(
    "method, path",
    [("get", "/predict"), ("put", "/predict"), ("get", "/predict/batch"), ("post", "/model-info"), ("delete", "/")],
)
def test_wrong_method_returns_405(client, method, path):
    assert getattr(client, method)(path).status_code == 405
