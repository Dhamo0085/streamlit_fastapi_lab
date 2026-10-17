"""End-to-end: Streamlit client helpers -> real FastAPI app (in-process).

FastAPI's TestClient has the same .get/.post interface as `requests`, so it
is passed as the `http` argument and the real app, validation and model are
exercised without starting a server. A fresh model is trained into tmp_path.
"""
import pytest
from fastapi.testclient import TestClient

import client
import predict
import train
from dashboard_fakes import DATA_DIR, SETOSA

pytestmark = pytest.mark.integration

BASE = "http://testserver"


@pytest.fixture(scope="module")
def trained_dir(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("integration_model")
    train.main(model_dir=model_dir)
    return model_dir


@pytest.fixture
def api(trained_dir, monkeypatch):
    from main import app

    monkeypatch.setattr(predict, "MODEL_PATH", trained_dir / "iris_model.pkl")
    predict.reset_model_cache()
    with TestClient(app) as test_client:
        yield test_client
    predict.reset_model_cache()


@pytest.fixture
def api_without_model(tmp_path, monkeypatch):
    from main import app

    monkeypatch.setattr(predict, "MODEL_PATH", tmp_path / "missing.pkl")
    predict.reset_model_cache()
    with TestClient(app) as test_client:
        yield test_client
    predict.reset_model_cache()


def test_check_backend_against_real_api(api):
    status = client.check_backend(BASE, http=api)
    assert status.online and status.model_loaded is True


def test_slider_payload_round_trip(api):
    payload = client.build_payload(5.1, 3.5, 1.4, 0.2)
    result = client.post_predict(payload, BASE, http=api)
    assert result.ok
    assert result.data == {"response": 0, "species": "setosa", "warnings": []}


def test_proba_round_trip(api):
    result = client.post_proba(client.build_payload(6.7, 3.0, 5.2, 2.3), BASE, http=api)
    assert result.ok
    assert result.data["species"] == "virginica"
    assert sum(result.data["probabilities"].values()) == pytest.approx(1.0)


def test_original_test_json_upload_round_trip(api):
    # data/test.json has sepal_length=2.5, below anything seen in training.
    upload = client.parse_uploaded_json((DATA_DIR / "test.json").read_bytes())
    result = client.post_proba(upload.payload, BASE, http=api)
    assert result.ok
    assert result.data["species"] in client.FLOWER_TYPES.values()
    assert any(w.startswith("sepal_length=2.5 is outside") for w in result.data["warnings"])


def test_sample_batch_csv_round_trip(api):
    df, _ = client.read_batch_csv((DATA_DIR / "sample_batch.csv").read_bytes())
    validation = client.validate_batch_df(df)
    result = client.post_batch(validation.frame.to_dict("records"), BASE, http=api)
    assert result.ok and result.data["count"] == 12
    table = client.batch_results_frame(validation.frame, result.data["predictions"])
    assert table["species"].value_counts().to_dict() == {"setosa": 4, "versicolor": 4, "virginica": 4}
    # Row 4 has sepal_width=4.4, above the training maximum of 4.1.
    assert table.loc[3, "warnings"].startswith("sepal_width=4.4 is outside")


def test_batch_larger_than_api_limit_is_chunked(api):
    result = client.post_batch([SETOSA] * 230, BASE, http=api)
    assert result.ok
    assert result.data["count"] == 230
    assert result.data["predictions"][-1]["index"] == 229


def test_server_side_validation_error_is_formatted(api):
    # Bypass client-side validation to prove the API's 422 is readable too.
    result = client.post_predict(SETOSA | {"sepal_length": -1}, BASE, http=api)
    assert result.status_code == 422
    assert result.error == "Invalid input - sepal_length: Input should be greater than 0"


def test_model_info_round_trip(api):
    result = client.get_model_info(BASE, http=api)
    assert result.ok
    assert result.data["metrics"]["n_test"] == 45


def test_missing_model_reaches_client_as_503(api_without_model):
    status = client.check_backend(BASE, http=api_without_model)
    assert status.online and status.model_loaded is False
    result = client.post_predict(SETOSA, BASE, http=api_without_model)
    assert result.status_code == 503
    assert "model is not available" in result.error
