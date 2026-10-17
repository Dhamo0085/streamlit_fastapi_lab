"""Unit tests for Streamlit_Labs/src/client.py (no browser, no server)."""
import io
import math

import pandas as pd
import pytest
import requests

import client
from dashboard_fakes import DATA_DIR, SETOSA, FakeResponse


# ------------------------------------------------------------------ get_backend_url

def test_backend_url_defaults_to_localhost():
    assert client.get_backend_url() == "http://localhost:8000"


def test_backend_url_from_env_var(monkeypatch):
    monkeypatch.setenv("FASTAPI_BACKEND_URL", "http://api.example:9000/")
    assert client.get_backend_url() == "http://api.example:9000"


def test_blank_env_var_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("FASTAPI_BACKEND_URL", "   ")
    assert client.get_backend_url() == "http://localhost:8000"


# ------------------------------------------------------------- build / validate payload

def test_build_payload_uses_original_field_names_and_floats():
    payload = client.build_payload(5, 3.5, 1.4, 0.2)
    assert payload == SETOSA | {"sepal_length": 5.0}
    assert all(isinstance(v, float) for v in payload.values())


def test_validate_payload_accepts_valid_input():
    assert client.validate_payload(SETOSA) == []


def test_validate_payload_accepts_limits():
    assert client.validate_payload({k: 10.0 for k in SETOSA}) == []
    assert client.validate_payload({k: 0.001 for k in SETOSA}) == []


@pytest.mark.parametrize(
    "payload, expected",
    [
        ([5.1, 3.5, 1.4, 0.2], "JSON object"),
        ({k: v for k, v in SETOSA.items() if k != "petal_width"}, "Missing field(s): petal_width"),
        (SETOSA | {"color": "blue"}, "Unknown field(s): color"),
        (SETOSA | {"sepal_length": "5.1"}, "sepal_length must be a number"),
        (SETOSA | {"sepal_length": None}, "sepal_length must be a number"),
        (SETOSA | {"sepal_length": True}, "sepal_length must be a number"),
        (SETOSA | {"sepal_width": math.nan}, "sepal_width must be a finite number"),
        (SETOSA | {"sepal_width": math.inf}, "sepal_width must be a finite number"),
        (SETOSA | {"petal_length": -1}, "petal_length must be > 0 and <= 10"),
        (SETOSA | {"petal_length": 0}, "petal_length must be > 0 and <= 10"),
        (SETOSA | {"petal_width": 10.5}, "petal_width must be > 0 and <= 10"),
    ],
    ids=["not-dict", "missing", "extra", "string", "none", "bool", "nan", "inf", "negative", "zero", "above-10"],
)
def test_validate_payload_rejects_bad_input(payload, expected):
    errors = client.validate_payload(payload)
    assert errors
    assert any(expected in e for e in errors), errors


# ------------------------------------------------------------------ parse_uploaded_json

def test_parse_original_test_json_file():
    with open(DATA_DIR / "test.json", "rb") as f:
        result = client.parse_uploaded_json(f)
    assert result.error is None
    assert result.payload == {"sepal_length": 2.5, "sepal_width": 3.5, "petal_length": 1.5, "petal_width": 2.5}
    assert result.data["input_test"] == result.payload


@pytest.mark.parametrize(
    "raw",
    [
        b'{"input_test": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}}',
        '{"input_test": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}}',
        io.BytesIO(b'{"input_test": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}}'),
        '﻿{"input_test": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}}'.encode("utf-8"),
    ],
    ids=["bytes", "str", "file-like", "utf8-bom"],
)
def test_parse_valid_json_inputs(raw):
    result = client.parse_uploaded_json(raw)
    assert result.error is None
    assert result.payload == SETOSA


@pytest.mark.parametrize(
    "raw, expected",
    [
        (b'{"input_test": {"sepal_length": 5.1,', "not valid JSON"),
        (b"", "not valid JSON"),
        (b'{"inputs": {}}', '"input_test"'),
        (b"[1, 2, 3]", '"input_test"'),
        (b'{"input_test": [5.1, 3.5, 1.4, 0.2]}', "Invalid input_test"),
        (b'{"input_test": {"sepal_length": "abc", "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}}',
         "sepal_length must be a number"),
        (b'{"input_test": {"sepal_length": -2, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}}',
         "sepal_length must be > 0"),
        (b'{"input_test": {"sepal_length": 5.1}}', "Missing field(s)"),
        ("café".encode("latin-1"), "not UTF-8"),
    ],
    ids=["malformed", "empty", "missing-key", "top-level-list", "input-not-object",
         "wrong-type", "negative", "missing-fields", "not-utf8"],
)
def test_parse_bad_json_returns_friendly_error(raw, expected):
    result = client.parse_uploaded_json(raw)
    assert result.payload is None
    assert expected in result.error


def test_parse_keeps_document_for_preview_when_key_missing():
    result = client.parse_uploaded_json(b'{"something": 1}')
    assert result.data == {"something": 1}
    assert result.error


# ------------------------------------------------------------------- batch CSV helpers

def _df(rows, columns=None):
    return pd.DataFrame(rows, columns=columns or client.FEATURE_COLUMNS)


def test_sample_batch_csv_is_valid():
    df, error = client.read_batch_csv((DATA_DIR / "sample_batch.csv").read_bytes())
    assert error is None
    validation = client.validate_batch_df(df)
    assert validation.ok
    assert len(validation.frame) == 12
    assert list(validation.frame.columns) == client.FEATURE_COLUMNS


def test_batch_missing_columns():
    validation = client.validate_batch_df(_df([[5.1, 3.5]], ["sepal_length", "sepal_width"]))
    assert not validation.ok
    assert "Missing column(s): petal_length, petal_width" in validation.errors[0]


def test_batch_extra_columns_are_dropped_with_note():
    df = _df([[1, 5.1, 3.5, 1.4, 0.2]], ["id"] + client.FEATURE_COLUMNS)
    validation = client.validate_batch_df(df)
    assert validation.ok
    assert list(validation.frame.columns) == client.FEATURE_COLUMNS
    assert validation.notes == ["Ignoring extra column(s): id."]


def test_batch_column_names_are_trimmed():
    df = _df([[5.1, 3.5, 1.4, 0.2]], [f" {c} " for c in client.FEATURE_COLUMNS])
    assert client.validate_batch_df(df).ok


def test_batch_nan_reports_row_number():
    validation = client.validate_batch_df(_df([[5.1, 3.5, 1.4, 0.2], [5.1, None, 1.4, 0.2]]))
    assert not validation.ok
    assert validation.errors == ["sepal_width: missing value (NaN) in row(s) 2."]


def test_batch_non_numeric_value():
    validation = client.validate_batch_df(_df([["abc", 3.5, 1.4, 0.2]]))
    assert validation.errors == ["sepal_length: non-numeric value in row(s) 1."]


@pytest.mark.parametrize("bad", [0, -1, 10.5, float("inf"), float("-inf")])
def test_batch_out_of_range(bad):
    validation = client.validate_batch_df(_df([[5.1, 3.5, 1.4, 0.2], [5.1, 3.5, bad, 0.2]]))
    assert validation.errors == ["petal_length: value outside (0, 10] cm in row(s) 2."]


def test_batch_empty():
    assert client.validate_batch_df(_df([])).errors == ["The CSV has no data rows."]
    assert client.validate_batch_df(None).errors == ["The CSV has no data rows."]


def test_batch_many_bad_rows_are_summarised():
    validation = client.validate_batch_df(_df([[-1, 3.5, 1.4, 0.2]] * 15))
    assert validation.errors[0].endswith("row(s) 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 (+5 more).")


def test_read_batch_csv_handles_empty_and_text():
    assert client.read_batch_csv(b"") == (None, "The CSV file is empty.")
    df, error = client.read_batch_csv("sepal_length,sepal_width,petal_length,petal_width\n5.1,3.5,1.4,0.2\n")
    assert error is None and len(df) == 1


def test_batch_results_frame():
    features = _df([[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]])
    predictions = [
        {"index": 1, "response": 2, "species": "virginica", "warnings": []},
        {"index": 0, "response": 0, "species": "setosa", "warnings": ["a", "b"]},
    ]
    results = client.batch_results_frame(features, predictions)
    assert results["species"].tolist() == ["setosa", "virginica"]
    assert results["warnings"].tolist() == ["a; b", ""]
    assert list(results.columns) == client.FEATURE_COLUMNS + ["response", "species", "warnings"]


# ---------------------------------------------------------------------- format_api_error

def test_format_422_single_field():
    body = {"detail": [{"loc": ["body", "sepal_length"], "msg": "Input should be greater than 0"}]}
    assert client.format_api_error(422, body) == "Invalid input - sepal_length: Input should be greater than 0"


def test_format_422_batch_uses_1_based_row():
    body = {"detail": [{"loc": ["body", "samples", 1, "petal_width"], "msg": "Input should be greater than 0"}]}
    assert client.format_api_error(422, body) == "Invalid input - row 2 petal_width: Input should be greater than 0"


def test_format_422_whole_body():
    body = {"detail": [{"loc": ["body"], "msg": "Field required"}]}
    assert client.format_api_error(422, body) == "Invalid input - request: Field required"


@pytest.mark.parametrize(
    "status, body, expected",
    [
        (503, {"detail": "Model file not found"}, "model is not available: Model file not found"),
        (404, {"detail": "Not Found"}, "Endpoint not found"),
        (500, {"detail": "boom"}, "HTTP 500: boom"),
        (502, None, "HTTP 502."),
        (500, "plain text", "HTTP 500: plain text"),
    ],
)
def test_format_other_errors(status, body, expected):
    assert expected in client.format_api_error(status, body)


# --------------------------------------------------------------------- HTTP helpers

def test_post_predict_success(fake_backend):
    fake_backend.set("POST /predict", FakeResponse(200, {"response": 0, "species": "setosa", "warnings": []}))
    result = client.post_predict(SETOSA)
    assert result.ok and result.data["species"] == "setosa"
    call = fake_backend.calls[0]
    assert call["url"] == "http://localhost:8000/predict"
    assert call["json"] == SETOSA
    assert call["timeout"] == client.REQUEST_TIMEOUT


def test_post_predict_uses_env_backend_url(fake_backend, monkeypatch):
    monkeypatch.setenv("FASTAPI_BACKEND_URL", "http://remote:1234")
    fake_backend.set("POST /predict", FakeResponse(200, {"response": 0}))
    client.post_predict(SETOSA)
    assert fake_backend.calls[0]["url"] == "http://remote:1234/predict"


def test_post_predict_422(fake_backend):
    body = {"detail": [{"loc": ["body", "sepal_length"], "msg": "Input should be greater than 0"}]}
    fake_backend.set("POST /predict", FakeResponse(422, body))
    result = client.post_predict(SETOSA | {"sepal_length": -1})
    assert not result.ok
    assert result.status_code == 422
    assert result.error == "Invalid input - sepal_length: Input should be greater than 0"


def test_post_proba_503(fake_backend):
    fake_backend.set("POST /predict/proba", FakeResponse(503, {"detail": "Model file not found"}))
    result = client.post_proba(SETOSA)
    assert result.status_code == 503
    assert "model is not available" in result.error


@pytest.mark.parametrize(
    "exc, expected",
    [
        (requests.Timeout("slow"), "did not answer within"),
        (requests.ConnectionError("refused"), "Could not connect"),
        (requests.RequestException("weird"), "Request to the backend failed"),
    ],
    ids=["timeout", "connection-error", "other-request-error"],
)
def test_network_errors_never_raise(fake_backend, exc, expected):
    fake_backend.set("POST /predict", exc)
    result = client.post_predict(SETOSA)
    assert not result.ok
    assert result.status_code is None
    assert expected in result.error


def test_non_json_success_body_is_an_error(fake_backend):
    fake_backend.set("POST /predict", FakeResponse(200, text_body="<html>"))
    result = client.post_predict(SETOSA)
    assert not result.ok and "not JSON" in result.error


def test_non_json_error_body(fake_backend):
    fake_backend.set("POST /predict", FakeResponse(500, text_body="Internal Server Error"))
    assert client.post_predict(SETOSA).error == "Backend returned HTTP 500."


def test_get_model_info(online_backend):
    result = client.get_model_info()
    assert result.ok and result.data["max_depth"] == 3


# -------------------------------------------------------------------- check_backend

def test_check_backend_online_with_model(online_backend):
    status = client.check_backend()
    assert status.online and status.model_loaded is True
    assert online_backend.calls[0]["timeout"] == client.HEALTH_TIMEOUT


def test_check_backend_online_without_model(fake_backend):
    fake_backend.set("GET /", FakeResponse(200, {"status": "healthy", "model_loaded": False}))
    status = client.check_backend()
    assert status.state == "online" and status.model_loaded is False


def test_check_backend_original_api_without_model_loaded_field(fake_backend):
    fake_backend.set("GET /", FakeResponse(200, {"status": "healthy"}))
    status = client.check_backend()
    assert status.online and status.model_loaded is None


def test_check_backend_offline(offline_backend):
    status = client.check_backend()
    assert status.state == "offline" and not status.online


def test_check_backend_timeout_counts_as_offline(fake_backend):
    fake_backend.set("GET /", requests.Timeout("slow"))
    assert client.check_backend().state == "offline"


def test_check_backend_problem_on_http_error(fake_backend):
    fake_backend.set("GET /", FakeResponse(500, {"detail": "oops"}))
    assert client.check_backend().state == "problem"


# ------------------------------------------------------------------------ post_batch

def _batch_handler(kwargs):
    samples = kwargs["json"]["samples"]
    assert 1 <= len(samples) <= 100
    return FakeResponse(200, {
        "count": len(samples),
        "predictions": [{"index": i, "response": 0, "species": "setosa", "warnings": []} for i in range(len(samples))],
    })


def test_post_batch_chunks_into_groups_of_100(fake_backend):
    fake_backend.set("POST /predict/batch", _batch_handler)
    result = client.post_batch([SETOSA] * 250)
    assert result.ok
    assert [len(c["json"]["samples"]) for c in fake_backend.calls] == [100, 100, 50]
    assert result.data["count"] == 250
    assert [p["index"] for p in result.data["predictions"]] == list(range(250))


def test_post_batch_single_chunk(fake_backend):
    fake_backend.set("POST /predict/batch", _batch_handler)
    assert client.post_batch([SETOSA] * 3).data["count"] == 3
    assert len(fake_backend.calls) == 1


def test_post_batch_reports_failing_chunk(fake_backend):
    responses = iter([_batch_handler, lambda kw: FakeResponse(503, {"detail": "gone"})])
    fake_backend.set("POST /predict/batch", lambda kw: next(responses)(kw))
    result = client.post_batch([SETOSA] * 150)
    assert not result.ok
    assert result.error.endswith("(in rows 101-150)")


def test_post_batch_empty():
    result = client.post_batch([])
    assert not result.ok and "no samples" in result.error
