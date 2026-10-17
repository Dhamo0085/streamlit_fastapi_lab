"""Pure helper functions used by Dashboard.py.

Everything here works without a browser or a running Streamlit server, so it
can be unit-tested. Functions never raise into the UI: network and parsing
problems come back as readable error messages instead.
"""
from __future__ import annotations

import io
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import requests

DEFAULT_BACKEND_URL = "http://localhost:8000"
BACKEND_URL_ENV_VAR = "FASTAPI_BACKEND_URL"

FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
FLOWER_TYPES = {0: "setosa", 1: "versicolor", 2: "virginica"}

# Mirrors the API's hard limits so obviously bad input is caught before a
# request is sent. The API still validates everything itself.
FEATURE_MIN_EXCLUSIVE = 0.0
FEATURE_MAX_INCLUSIVE = 10.0
MAX_BATCH_SIZE = 100

HEALTH_TIMEOUT = 5
REQUEST_TIMEOUT = 10


@dataclass
class ApiResult:
    """Outcome of one API call. Exactly one of `data` / `error` is set."""

    ok: bool
    status_code: int | None = None
    data: Any = None
    error: str | None = None


@dataclass
class BackendStatus:
    """Backend health as shown in the sidebar: online, offline or problem."""

    state: str
    model_loaded: bool | None = None
    message: str = ""

    @property
    def online(self) -> bool:
        return self.state == "online"


@dataclass
class UploadResult:
    """Parsed JSON upload: the whole document plus its `input_test` payload."""

    data: Any = None
    payload: dict | None = None
    error: str | None = None


@dataclass
class BatchValidation:
    """Cleaned batch DataFrame (only the 4 feature columns) plus any problems."""

    frame: pd.DataFrame | None
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.frame is not None and not self.errors


# ----------------------------------------------------------------------------- config

def get_backend_url() -> str:
    """Backend base URL from $FASTAPI_BACKEND_URL, defaulting to localhost:8000."""
    url = os.environ.get(BACKEND_URL_ENV_VAR, "").strip()
    return (url or DEFAULT_BACKEND_URL).rstrip("/")


# ------------------------------------------------------------------------- validation

def _is_number(value: Any) -> bool:
    # bool is a subclass of int, but True is not a measurement.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def build_payload(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float) -> dict:
    """Request body for /predict, using the original API field names."""
    return {
        "sepal_length": float(sepal_length),
        "sepal_width": float(sepal_width),
        "petal_length": float(petal_length),
        "petal_width": float(petal_width),
    }


def validate_payload(payload: Any) -> list[str]:
    """Return a list of human-readable problems; an empty list means valid."""
    if not isinstance(payload, dict):
        return ["Input must be a JSON object with the four iris measurements."]

    errors = []
    missing = [name for name in FEATURE_COLUMNS if name not in payload]
    if missing:
        errors.append(f"Missing field(s): {', '.join(missing)}.")
    extra = sorted(set(payload) - set(FEATURE_COLUMNS))
    if extra:
        errors.append(f"Unknown field(s): {', '.join(extra)}.")

    for name in FEATURE_COLUMNS:
        if name not in payload:
            continue
        value = payload[name]
        if not _is_number(value):
            errors.append(f"{name} must be a number (got {value!r}).")
        elif not math.isfinite(value):
            errors.append(f"{name} must be a finite number (got {value!r}).")
        elif not FEATURE_MIN_EXCLUSIVE < value <= FEATURE_MAX_INCLUSIVE:
            errors.append(f"{name} must be > 0 and <= 10 cm (got {value:g}).")
    return errors


def parse_uploaded_json(uploaded: Any) -> UploadResult:
    """
    Parse a JSON upload in the format of data/test.json:
    {"input_test": {"sepal_length": ..., "sepal_width": ..., ...}}

    Accepts bytes, str, or a file-like object (Streamlit's UploadedFile).
    """
    try:
        raw = uploaded.read() if hasattr(uploaded, "read") else uploaded
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8-sig")
        data = json.loads(raw)
    except UnicodeDecodeError:
        return UploadResult(error="The file is not UTF-8 text. Please upload a JSON file.")
    except json.JSONDecodeError as exc:
        return UploadResult(error=f"The file is not valid JSON (line {exc.lineno}, column {exc.colno}: {exc.msg}).")
    except Exception as exc:  # pragma: no cover - defensive: never crash the page
        return UploadResult(error=f"Could not read the file: {exc}")

    if not isinstance(data, dict) or "input_test" not in data:
        return UploadResult(
            data=data,
            error='The JSON must contain an "input_test" object, like data/test.json.',
        )
    payload = data["input_test"]
    problems = validate_payload(payload)
    if problems:
        return UploadResult(data=data, error="Invalid input_test: " + " ".join(problems))
    return UploadResult(data=data, payload=payload)


def validate_batch_df(df: pd.DataFrame) -> BatchValidation:
    """
    Check an uploaded CSV before sending it to /predict/batch.

    Extra columns (e.g. an id column) are dropped with a note rather than
    rejected; missing, non-numeric, NaN and out-of-range values are errors.
    Row numbers in messages are 1-based data rows (header not counted).
    """
    if df is None or df.empty:
        return BatchValidation(None, ["The CSV has no data rows."])

    df = df.rename(columns=lambda c: str(c).strip())
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        return BatchValidation(
            None,
            [f"Missing column(s): {', '.join(missing)}. Expected: {', '.join(FEATURE_COLUMNS)}."],
        )

    notes = []
    extra = [c for c in df.columns if c not in FEATURE_COLUMNS]
    if extra:
        notes.append(f"Ignoring extra column(s): {', '.join(extra)}.")

    errors = []
    clean = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    for column in FEATURE_COLUMNS:
        values = clean[column]
        not_numeric = values.isna() & df[column].notna()
        if not_numeric.any():
            errors.append(f"{column}: non-numeric value in row(s) {_rows(not_numeric)}.")
        empty = df[column].isna()
        if empty.any():
            errors.append(f"{column}: missing value (NaN) in row(s) {_rows(empty)}.")
        # +/-inf also lands here, since inf > 10 and -inf <= 0.
        out_of_range = values.notna() & ((values <= FEATURE_MIN_EXCLUSIVE) | (values > FEATURE_MAX_INCLUSIVE))
        if out_of_range.any():
            errors.append(f"{column}: value outside (0, 10] cm in row(s) {_rows(out_of_range)}.")

    return BatchValidation(clean if not errors else None, errors, notes)


def _rows(mask: pd.Series, limit: int = 10) -> str:
    rows = [str(i + 1) for i, flag in enumerate(mask.tolist()) if flag]
    shown = ", ".join(rows[:limit])
    return shown + (f" (+{len(rows) - limit} more)" if len(rows) > limit else "")


def read_batch_csv(uploaded: Any) -> tuple[pd.DataFrame | None, str | None]:
    """Read an uploaded CSV into a DataFrame, returning (df, error)."""
    try:
        raw = uploaded.read() if hasattr(uploaded, "read") else uploaded
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        return pd.read_csv(io.BytesIO(raw)), None
    except pd.errors.EmptyDataError:
        return None, "The CSV file is empty."
    except Exception as exc:
        return None, f"Could not read the CSV file: {exc}"


# ----------------------------------------------------------------------------- errors

def format_api_error(status_code: int | None, body: Any) -> str:
    """Turn an API error response into one readable sentence for the UI."""
    detail = body.get("detail") if isinstance(body, dict) else body

    if status_code == 422 and isinstance(detail, list):
        parts = []
        for err in detail:
            loc = [str(p) for p in err.get("loc", []) if p != "body"]
            if len(loc) >= 2 and loc[0] == "samples" and loc[1].isdigit():
                loc = [f"row {int(loc[1]) + 1}"] + loc[2:]
            where = " ".join(loc) or "request"
            parts.append(f"{where}: {err.get('msg', 'invalid value')}")
        return "Invalid input - " + "; ".join(parts)
    if status_code == 503:
        return f"The backend is running but the model is not available: {detail or 'run train.py'}"
    if status_code == 404:
        return "Endpoint not found on the backend (is it the updated FastAPI app?)."
    if detail:
        return f"Backend returned HTTP {status_code}: {detail}"
    return f"Backend returned HTTP {status_code}."


# ------------------------------------------------------------------------------- HTTP

def _request(method: str, path: str, *, payload: Any = None, base_url: str | None = None,
             timeout: float = REQUEST_TIMEOUT, http: Any = None) -> ApiResult:
    """
    Call the backend and wrap the outcome in an ApiResult.

    `http` defaults to the requests module; tests pass FastAPI's TestClient
    (same .get/.post interface) to run against the real app in-process.
    """
    http = http or requests
    base_url = (base_url or get_backend_url()).rstrip("/")
    url = f"{base_url}{path}"
    kwargs = {"timeout": timeout}
    if payload is not None:
        kwargs["json"] = payload
    try:
        response = getattr(http, method)(url, **kwargs)
    except requests.Timeout:
        return ApiResult(False, error=f"The backend at {base_url} did not answer within {timeout} s.")
    except requests.ConnectionError:
        return ApiResult(False, error=f"Could not connect to the backend at {base_url}. Is uvicorn running?")
    except requests.RequestException as exc:
        return ApiResult(False, error=f"Request to the backend failed: {exc}")

    try:
        body = response.json()
    except ValueError:
        body = None

    if response.status_code == 200 and body is not None:
        return ApiResult(True, 200, data=body)
    if response.status_code == 200:
        return ApiResult(False, 200, error="The backend returned a response that is not JSON.")
    return ApiResult(False, response.status_code, data=body, error=format_api_error(response.status_code, body))


def check_backend(base_url: str | None = None, timeout: float = HEALTH_TIMEOUT, http: Any = None) -> BackendStatus:
    """
    Ask the backend (GET /) whether it is up and whether its model is loaded.
    This replaces the original check for a model file on the local disk.
    """
    result = _request("get", "/", base_url=base_url, timeout=timeout, http=http)
    if result.ok and isinstance(result.data, dict):
        return BackendStatus("online", result.data.get("model_loaded"), "Backend online")
    if result.status_code is None:
        return BackendStatus("offline", None, result.error or "Backend offline")
    return BackendStatus("problem", None, result.error or "Problem connecting")


def post_predict(payload: dict, base_url: str | None = None, timeout: float = REQUEST_TIMEOUT, http: Any = None) -> ApiResult:
    """POST /predict (the original endpoint)."""
    return _request("post", "/predict", payload=payload, base_url=base_url, timeout=timeout, http=http)


def post_proba(payload: dict, base_url: str | None = None, timeout: float = REQUEST_TIMEOUT, http: Any = None) -> ApiResult:
    """POST /predict/proba: species plus class probabilities."""
    return _request("post", "/predict/proba", payload=payload, base_url=base_url, timeout=timeout, http=http)


def get_model_info(base_url: str | None = None, timeout: float = REQUEST_TIMEOUT, http: Any = None) -> ApiResult:
    """GET /model-info."""
    return _request("get", "/model-info", base_url=base_url, timeout=timeout, http=http)


def post_batch(samples: list[dict], base_url: str | None = None, timeout: float = REQUEST_TIMEOUT,
               http: Any = None, chunk_size: int = MAX_BATCH_SIZE) -> ApiResult:
    """
    POST /predict/batch in chunks of at most 100 samples (the API limit).

    Returns one combined result whose prediction indices refer to the
    position in `samples`. Stops at the first failing chunk.
    """
    if not samples:
        return ApiResult(False, error="There are no samples to predict.")

    predictions = []
    for start in range(0, len(samples), chunk_size):
        chunk = samples[start:start + chunk_size]
        result = _request("post", "/predict/batch", payload={"samples": chunk},
                          base_url=base_url, timeout=timeout, http=http)
        if not result.ok:
            if start:
                result.error = f"{result.error} (in rows {start + 1}-{start + len(chunk)})"
            return result
        for item in result.data["predictions"]:
            predictions.append({**item, "index": item["index"] + start})
    return ApiResult(True, 200, data={"count": len(predictions), "predictions": predictions})


def batch_results_frame(features: pd.DataFrame, predictions: list[dict]) -> pd.DataFrame:
    """Join input rows with their predictions into one table for display/download."""
    results = features.reset_index(drop=True).copy()
    by_index = {p["index"]: p for p in predictions}
    results["response"] = [by_index[i]["response"] for i in range(len(results))]
    results["species"] = [by_index[i]["species"] for i in range(len(results))]
    results["warnings"] = ["; ".join(by_index[i].get("warnings", [])) for i in range(len(results))]
    return results
