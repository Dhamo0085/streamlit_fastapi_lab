import streamlit as st
from streamlit.logger import get_logger

from client import (
    FLOWER_TYPES,
    build_payload,
    check_backend,
    get_backend_url,
    parse_uploaded_json,
    post_predict,
    validate_payload,
)

# Configurable via the FASTAPI_BACKEND_URL environment variable (B11).
FASTAPI_BACKEND_ENDPOINT = get_backend_url()

LOGGER = get_logger(__name__)


def predict_flower(client_input: dict) -> None:
    """Send prediction request and display result."""
    result_container = st.empty()
    problems = validate_payload(client_input)
    if problems:
        result_container.error("Cannot predict: " + " ".join(problems))
        return

    with st.spinner('Predicting...'):
        result = post_predict(client_input, base_url=FASTAPI_BACKEND_ENDPOINT)

    if result.ok:
        flower_class = result.data.get("response")
        if flower_class in FLOWER_TYPES:
            result_container.success(f"The flower predicted is: {FLOWER_TYPES[flower_class]}")
        else:
            result_container.error("Invalid prediction response")
            LOGGER.error(f"Unexpected response: {flower_class}")
    else:
        LOGGER.error(f"Prediction error: {result.error}")
        st.toast(f':red[{result.error}]', icon="🔴")
        result_container.error(result.error)


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
            upload = parse_uploaded_json(test_input_file)
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

    st.write("# Iris Flower Prediction! 🪻")

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
            else:
                client_input = build_payload(sepal_length, sepal_width, petal_length, petal_width)
            predict_flower(client_input)

if __name__ == "__main__":
    run()
