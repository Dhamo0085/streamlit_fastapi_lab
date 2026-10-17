"""Smoke tests for Dashboard.py using Streamlit's AppTest (headless, no browser).

The backend is faked by patching requests.get/requests.post (see conftest.py).
These tests check that the page renders and reacts correctly; they do not
check visual layout.
"""
import pytest
from streamlit.testing.v1 import AppTest

from dashboard_fakes import DASHBOARD, FakeResponse


def run_app() -> AppTest:
    return AppTest.from_file(str(DASHBOARD), default_timeout=30).run()


def texts(elements) -> list[str]:
    return [e.value for e in elements]


def predict_calls(backend) -> list[dict]:
    # The Model info tab also calls the backend on every run, so filter.
    return [c for c in backend.calls if c["path"].startswith("/predict")]


def test_renders_with_backend_online(online_backend):
    at = run_app()
    assert not at.exception
    assert texts(at.sidebar.success) == ["Backend online ✅"]
    assert not at.sidebar.warning  # B7: no stray warning next to the success box
    assert [t.label for t in at.tabs] == ["Predict", "Batch", "Model info", "History"]


def test_renders_with_backend_offline(offline_backend):
    at = run_app()
    assert not at.exception
    assert texts(at.sidebar.error) == ["Backend offline 😱"]
    assert "Backend offline" in texts(at.tabs[2].warning)[0]


def test_backend_problem_shows_warning(fake_backend):
    fake_backend.set("GET /", FakeResponse(500, {"detail": "oops"}))
    at = run_app()
    assert not at.exception
    assert texts(at.sidebar.warning) == ["Problem connecting 😭"]
    assert not at.sidebar.success


def test_original_sliders_are_kept():
    at = AppTest.from_file(str(DASHBOARD), default_timeout=30)
    at.run()
    sliders = {s.label: (s.min, s.max, s.value, s.step) for s in at.sidebar.slider}
    assert sliders == {
        "Sepal Length": (4.3, 7.9, 4.3, 0.1),
        "Sepal Width": (2.0, 4.4, 2.0, 0.1),
        "Petal Length": (1.0, 6.9, 1.0, 0.1),
        "Petal Width": (0.1, 2.5, 0.1, 0.1),
    }
    assert [b.label for b in at.sidebar.button] == ["Predict"]
    assert at.session_state["IS_JSON_FILE_AVAILABLE"] is False


def test_predict_shows_species_chart_and_history(online_backend):
    at = run_app()
    at.sidebar.button[0].click().run()
    assert not at.exception
    predict_tab, _, _, history_tab = at.tabs
    assert texts(predict_tab.success) == ["The flower predicted is: setosa"]
    assert len(history_tab.dataframe) == 1
    assert len(history_tab.dataframe[0].value) == 1
    # The request used the slider values and the proba endpoint.
    (call,) = predict_calls(online_backend)
    assert call["path"] == "/predict/proba"
    assert call["json"] == {"sepal_length": 4.3, "sepal_width": 2.0, "petal_length": 1.0, "petal_width": 0.1}


def test_predict_shows_api_warnings(online_backend):
    online_backend.set("POST /predict/proba", FakeResponse(200, {
        "response": 0, "species": "setosa",
        "probabilities": {"setosa": 1.0, "versicolor": 0.0, "virginica": 0.0},
        "warnings": ["sepal_width=4.4 is outside the range seen in training (2.0-4.1); the prediction may be unreliable"],
    }))
    at = run_app()
    at.sidebar.button[0].click().run()
    assert texts(at.tabs[0].warning) == [
        "sepal_width=4.4 is outside the range seen in training (2.0-4.1); the prediction may be unreliable"
    ]


def test_predict_falls_back_to_original_endpoint(online_backend):
    online_backend.set("POST /predict/proba", FakeResponse(404, {"detail": "Not Found"}))
    at = run_app()
    at.sidebar.button[0].click().run()
    assert texts(at.tabs[0].success) == ["The flower predicted is: setosa"]
    assert [c["path"] for c in predict_calls(online_backend)] == ["/predict/proba", "/predict"]


def test_predict_error_is_shown_not_raised(online_backend):
    online_backend.set("POST /predict/proba", FakeResponse(503, {"detail": "Model file not found"}))
    at = run_app()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert "model is not available" in texts(at.tabs[0].error)[0]
    assert any("model is not available" in t for t in texts(at.toast))


def test_unexpected_class_id_is_reported(online_backend):
    online_backend.set("POST /predict/proba", FakeResponse(200, {"response": 7, "species": "?", "warnings": []}))
    at = run_app()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert texts(at.tabs[0].error) == ["Invalid prediction response"]
    assert len(at.tabs[3].dataframe) == 0  # nothing added to history


def test_predict_when_offline_shows_toast(offline_backend):
    at = run_app()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert any("Backend offline" in t for t in texts(at.toast))


def test_model_not_loaded_uses_backend_status_not_local_file(fake_backend):
    # B8: the backend says the model is missing -> warn, and do not call /predict.
    fake_backend.set("GET /", FakeResponse(200, {"status": "healthy", "model_loaded": False}))
    at = run_app()
    assert "Model not loaded" in texts(at.sidebar.warning)[0]
    at.sidebar.button[0].click().run()
    assert any("Model not found. Run train.py" in t for t in texts(at.toast))
    assert predict_calls(fake_backend) == []


def test_model_info_tab(online_backend):
    at = run_app()
    metrics = {m.label: m.value for m in at.tabs[2].metric}
    assert metrics == {"Test accuracy": "95.6%", "5-fold CV accuracy": "94.3%", "Train / test rows": "105 / 45"}
    assert len(at.tabs[2].dataframe) == 2  # confusion matrix + training ranges


def test_model_info_without_metrics(online_backend):
    online_backend.set("GET /model-info", FakeResponse(200, {
        "model_type": "DecisionTreeClassifier", "max_depth": 3,
        "classes": ["setosa", "versicolor", "virginica"],
        "feature_names": [], "metrics_available": False, "metrics": None,
    }))
    at = run_app()
    assert "No metrics.json" in texts(at.tabs[2].info)[0]


def test_clear_history(online_backend):
    at = run_app()
    at.sidebar.button[0].click().run()
    at.sidebar.button[0].click().run()
    assert len(at.tabs[3].dataframe[0].value) == 2
    at.tabs[3].button[0].click().run()
    assert not at.exception
    assert texts(at.tabs[3].info) == ["No predictions yet in this session."]
    assert at.session_state["history"] == []


def test_backend_url_env_var_is_used(online_backend, monkeypatch):
    monkeypatch.setenv("FASTAPI_BACKEND_URL", "http://iris-api:9000")
    at = run_app()
    assert online_backend.calls[0]["url"] == "http://iris-api:9000/"
    assert "Backend: http://iris-api:9000" in texts(at.sidebar.caption)


@pytest.mark.skip(reason="streamlit 1.37 AppTest cannot drive st.file_uploader; "
                         "JSON parsing/errors are covered in test_client.py::test_parse_*")
def test_json_upload_overrides_sliders():
    pass


@pytest.mark.skip(reason="streamlit 1.37 AppTest cannot drive st.file_uploader; "
                         "CSV validation/batching is covered in test_client.py and test_integration.py")
def test_batch_csv_upload_and_download():
    pass
