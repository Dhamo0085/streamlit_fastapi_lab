from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.logger import get_logger

from client import (
    FEATURE_COLUMNS,
    FLOWER_TYPES,
    batch_results_frame,
    build_payload,
    check_backend,
    get_backend_url,
    get_model_info,
    parse_uploaded_json,
    post_batch,
    post_predict,
    post_proba,
    read_batch_csv,
    validate_batch_df,
    validate_payload,
)

# Configurable via the FASTAPI_BACKEND_URL environment variable (B11).
FASTAPI_BACKEND_ENDPOINT = get_backend_url()
SAMPLE_BATCH_CSV = Path(__file__).resolve().parents[1] / "data" / "sample_batch.csv"

LOGGER = get_logger(__name__)


# ------------------------------------------------------------------- single prediction

def add_to_history(client_input: dict, prediction: dict, source: str) -> None:
    """Append one prediction to this browser session's history table."""
    probabilities = prediction.get("probabilities") or {}
    st.session_state.setdefault("history", []).append(
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "source": source,
            **{name: client_input[name] for name in FEATURE_COLUMNS},
            "species": prediction["species"],
            "confidence": max(probabilities.values()) if probabilities else None,
            "warnings": len(prediction.get("warnings", [])),
        }
    )


def predict_flower(client_input: dict, source: str = "sliders") -> None:
    """Send prediction request and store the result for display."""
    result_container = st.empty()
    problems = validate_payload(client_input)
    if problems:
        st.session_state.pop("last_prediction", None)
        result_container.error("Cannot predict: " + " ".join(problems))
        return

    with st.spinner('Predicting...'):
        result = post_proba(client_input, base_url=FASTAPI_BACKEND_ENDPOINT)
        if result.status_code == 404:
            # Older backend without /predict/proba: fall back to the original endpoint.
            result = post_predict(client_input, base_url=FASTAPI_BACKEND_ENDPOINT)

    if result.ok:
        flower_class = result.data.get("response")
        if flower_class in FLOWER_TYPES:
            prediction = {**result.data, "species": FLOWER_TYPES[flower_class]}
            st.session_state["last_prediction"] = {"input": client_input, "source": source, **prediction}
            add_to_history(client_input, prediction, source)
            st.toast(f"Predicted: {prediction['species']}", icon="🪻")
        else:
            st.session_state.pop("last_prediction", None)
            result_container.error("Invalid prediction response")
            LOGGER.error(f"Unexpected response: {flower_class}")
    else:
        st.session_state.pop("last_prediction", None)
        LOGGER.error(f"Prediction error: {result.error}")
        st.toast(f':red[{result.error}]', icon="🔴")
        result_container.error(result.error)


def render_prediction(prediction: dict) -> None:
    """Species message, probability chart and extrapolation warnings."""
    st.success(f"The flower predicted is: {prediction['species']}")
    inputs = ", ".join(f"{name}={prediction['input'][name]:g}" for name in FEATURE_COLUMNS)
    st.caption(f"Input from {prediction['source']}: {inputs}")

    for warning in prediction.get("warnings", []):
        st.warning(warning, icon="⚠️")

    probabilities = prediction.get("probabilities")
    if probabilities:
        st.write("**Class probabilities**")
        chart = pd.DataFrame({"probability": probabilities})
        st.bar_chart(chart, x_label="species", y_label="probability")


# ------------------------------------------------------------------------------- tabs

def render_predict_tab(predict_button: bool, backend_status, slider_payload: dict) -> None:
    if predict_button:
        # B8: ask the backend whether its model is loaded instead of looking
        # for the .pkl file in a sibling folder on this machine.
        if not backend_status.online:
            st.toast(':red[Backend offline. Start uvicorn and refresh]', icon="🔴")
        elif backend_status.model_loaded is False:
            LOGGER.warning('Backend reports model not loaded')
            st.toast(':red[Model not found. Run train.py]', icon="🔥")
        else:
            if st.session_state.get("IS_JSON_FILE_AVAILABLE"):
                client_input = st.session_state["test_input_data"]['input_test']
                st.info("Using values from the uploaded JSON file (sliders ignored).")
                predict_flower(client_input, source="JSON file")
            else:
                predict_flower(slider_payload, source="sliders")

    if st.session_state.get("last_prediction"):
        render_prediction(st.session_state["last_prediction"])
    elif not predict_button:
        st.write("Set the measurements in the sidebar (or upload a JSON file) and press **Predict**.")


def render_batch_tab(backend_status) -> None:
    st.write(
        f"Upload a CSV with the columns `{', '.join(FEATURE_COLUMNS)}` (cm, each > 0 and <= 10). "
        "Rows are sent to `/predict/batch` in chunks of 100."
    )
    if SAMPLE_BATCH_CSV.is_file():
        st.download_button("Download sample CSV", SAMPLE_BATCH_CSV.read_bytes(),
                           file_name="sample_batch.csv", mime="text/csv")

    batch_file = st.file_uploader("Upload batch CSV", type=["csv"], key="batch_csv")
    if not batch_file:
        st.session_state.pop("batch_results", None)
        return

    # Results survive reruns (e.g. clicking download) but not a new file.
    file_key = f"{batch_file.name}:{batch_file.size}"
    if st.session_state.get("batch_file_key") != file_key:
        st.session_state["batch_file_key"] = file_key
        st.session_state.pop("batch_results", None)

    df, read_error = read_batch_csv(batch_file.getvalue())
    if read_error:
        st.error(read_error)
        return

    validation = validate_batch_df(df)
    for note in validation.notes:
        st.info(note)
    for error in validation.errors:
        st.error(error)
    st.write(f"Preview ({len(df)} rows)")
    st.dataframe(df.head(20))
    if not validation.ok:
        return

    if st.button("Run batch prediction", disabled=not backend_status.online):
        with st.spinner(f"Predicting {len(validation.frame)} rows..."):
            result = post_batch(validation.frame.to_dict("records"), base_url=FASTAPI_BACKEND_ENDPOINT)
        if result.ok:
            st.session_state["batch_results"] = batch_results_frame(validation.frame, result.data["predictions"])
        else:
            LOGGER.error(f"Batch error: {result.error}")
            st.error(result.error)

    results = st.session_state.get("batch_results")
    if results is not None:
        counts = results["species"].value_counts().to_dict()
        summary = ", ".join(f"{species}: {n}" for species, n in counts.items())
        st.success(f"Predicted {len(results)} rows ({summary}).")
        n_warned = int((results["warnings"] != "").sum())
        if n_warned:
            st.warning(f"{n_warned} row(s) are outside the training range; see the warnings column.")
        st.dataframe(results)
        st.download_button("Download results as CSV", results.to_csv(index=False).encode("utf-8"),
                           file_name="iris_predictions.csv", mime="text/csv")


def render_model_info_tab(backend_status) -> None:
    if not backend_status.online:
        st.warning("Backend offline: model information is unavailable.")
        return
    result = get_model_info(base_url=FASTAPI_BACKEND_ENDPOINT)
    if not result.ok:
        st.error(result.error)
        return

    info = result.data
    st.write(
        f"**{info['model_type']}** (max_depth={info['max_depth']}) · "
        f"classes: {', '.join(info['classes'])}"
    )
    metrics = info.get("metrics")
    if not metrics:
        st.info("No metrics.json found on the backend. Run train.py to generate it.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Test accuracy", f"{metrics['test_accuracy']:.1%}")
    col2.metric(f"{metrics['cv_folds']}-fold CV accuracy", f"{metrics['cv_accuracy_mean']:.1%}",
                help=f"± {metrics['cv_accuracy_std']:.1%} std across folds (training split)")
    col3.metric("Train / test rows", f"{metrics['n_train']} / {metrics['n_test']}")

    names = metrics["target_names"]
    st.write("**Confusion matrix (test set)**")
    st.dataframe(pd.DataFrame(metrics["confusion_matrix"],
                              index=[f"actual {n}" for n in names],
                              columns=[f"predicted {n}" for n in names]))

    st.write("**Feature importance**")
    importances = pd.DataFrame({"importance": metrics["feature_importances"]})
    st.bar_chart(importances, x_label="feature", y_label="importance")

    st.write("**Training feature ranges** (inputs outside these get a warning)")
    st.dataframe(pd.DataFrame(metrics["feature_ranges"]).T)
    st.caption(f"Trained {metrics['trained_at']} with scikit-learn {metrics['sklearn_version']}.")


def render_history_tab() -> None:
    history = st.session_state.setdefault("history", [])
    if history:
        st.dataframe(pd.DataFrame(history))
    else:
        st.info("No predictions yet in this session.")
    if st.button("Clear history", disabled=not history):
        st.session_state["history"] = []
        st.session_state.pop("last_prediction", None)
        st.rerun()


# -------------------------------------------------------------------------------- page

def render_backend_status():
    """Sidebar indicator. Returns the BackendStatus so callers can reuse it."""
    status = check_backend(FASTAPI_BACKEND_ENDPOINT)
    if status.state == "online":
        # B7: one box per state (the original nested st.warning inside st.success).
        st.success("Backend online ✅")
        if status.model_loaded is False:
            st.warning("Model not loaded on the backend. Run train.py 🔥")
    elif status.state == "problem":
        LOGGER.error(f"Backend problem: {status.message}")
        st.warning("Problem connecting 😭")
    else:
        LOGGER.error(f"Backend error: {status.message}")
        st.error("Backend offline 😱")
    return status


def run():
    st.set_page_config(page_title="Iris Flower Prediction Demo", page_icon="🪻")

    with st.sidebar:
        backend_status = render_backend_status()

        st.info("Configure parameters")

        sepal_length = st.slider("Sepal Length", 4.3, 7.9, 4.3, 0.1, help="cm")
        sepal_width = st.slider("Sepal Width", 2.0, 4.4, 2.0, 0.1, help="cm")
        petal_length = st.slider("Petal Length", 1.0, 6.9, 1.0, 0.1, help="cm")
        petal_width = st.slider("Petal Width", 0.1, 2.5, 0.1, 0.1, help="cm")

        test_input_file = st.file_uploader('Upload test prediction file', type=['json'])

        st.session_state["IS_JSON_FILE_AVAILABLE"] = False
        if test_input_file:
            # B9: malformed JSON or a missing "input_test" key shows an error
            # instead of crashing the page.
            upload = parse_uploaded_json(test_input_file.getvalue())
            if upload.data is not None:
                st.write('Preview file')
                st.json(upload.data)
            if upload.error:
                st.error(upload.error)
            else:
                st.session_state["IS_JSON_FILE_AVAILABLE"] = True
                st.session_state["test_input_data"] = upload.data
                st.caption("ℹ️ Values from the uploaded file will be used instead of the sliders.")

        predict_button = st.button('Predict')
        st.caption(f"Backend: {FASTAPI_BACKEND_ENDPOINT}")

    st.write("# Iris Flower Prediction! 🪻")

    tab_predict, tab_batch, tab_info, tab_history = st.tabs(["Predict", "Batch", "Model info", "History"])
    with tab_predict:
        slider_payload = build_payload(sepal_length, sepal_width, petal_length, petal_width)
        render_predict_tab(predict_button, backend_status, slider_payload)
    with tab_batch:
        render_batch_tab(backend_status)
    with tab_info:
        render_model_info_tab(backend_status)
    with tab_history:
        render_history_tab()

if __name__ == "__main__":
    run()
