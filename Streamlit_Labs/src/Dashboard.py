import json
import requests
import streamlit as st
from pathlib import Path
from streamlit.logger import get_logger

FASTAPI_BACKEND_ENDPOINT = "http://localhost:8000"
FASTAPI_IRIS_MODEL_LOCATION = Path(__file__).resolve().parents[2] / 'FastAPI_Labs' / 'model' / 'iris_model.pkl'

FLOWER_TYPES = {0: "setosa", 1: "versicolor", 2: "virginica"}
LOGGER = get_logger(__name__)

def predict_flower(client_input: dict) -> None:
    """Send prediction request and display result."""
    result_container = st.empty()
    try:
        with st.spinner('Predicting...'):
            response = requests.post(
                f'{FASTAPI_BACKEND_ENDPOINT}/predict',
                json=client_input,
                timeout=10
            )
        
        if response.status_code == 200:
            iris_content = response.json()
            flower_class = iris_content.get("response")
            if flower_class in FLOWER_TYPES:
                result_container.success(f"The flower predicted is: {FLOWER_TYPES[flower_class]}")
            else:
                result_container.error("Invalid prediction response")
                LOGGER.error(f"Unexpected response: {flower_class}")
        else:
            st.toast(f':red[Status: {response.status_code}. Check backend]', icon="🔴")
    except Exception as e:
        st.toast(':red[Backend error. Refresh and retry]', icon="🔴")
        LOGGER.error(f"Prediction error: {e}")

def run():
    st.set_page_config(page_title="Iris Flower Prediction Demo", page_icon="🪻")

    with st.sidebar:
        # Backend status check
        try:
            backend_request = requests.get(FASTAPI_BACKEND_ENDPOINT, timeout=5)
            st.success("Backend online ✅" if backend_request.status_code == 200 else st.warning("Problem connecting 😭"))
        except requests.ConnectionError as ce:
            LOGGER.error(f"Backend error: {ce}")
            st.error("Backend offline 😱")

        st.info("Configure parameters")
        
        sepal_length = st.slider("Sepal Length", 4.3, 7.9, 4.3, 0.1, help="cm")
        sepal_width = st.slider("Sepal Width", 2.0, 4.4, 2.0, 0.1, help="cm")
        petal_length = st.slider("Petal Length", 1.0, 6.9, 1.0, 0.1, help="cm")
        petal_width = st.slider("Petal Width", 0.1, 2.5, 0.1, 0.1, help="cm")
        
        test_input_file = st.file_uploader('Upload test prediction file', type=['json'])
        
        if test_input_file:
            st.write('Preview file')
            test_input_data = json.load(test_input_file)
            st.json(test_input_data)
            st.session_state["IS_JSON_FILE_AVAILABLE"] = True
            st.session_state["test_input_data"] = test_input_data
        else:
            st.session_state["IS_JSON_FILE_AVAILABLE"] = False
        
        predict_button = st.button('Predict')

    st.write("# Iris Flower Prediction! 🪻")
    
    if predict_button:
        if FASTAPI_IRIS_MODEL_LOCATION.is_file():
            if st.session_state.get("IS_JSON_FILE_AVAILABLE"):
                client_input = st.session_state["test_input_data"]['input_test']
            else:
                client_input = {
                    "petal_length": petal_length,
                    "sepal_length": sepal_length,
                    "petal_width": petal_width,
                    "sepal_width": sepal_width
                }
            predict_flower(client_input)
        else:
            LOGGER.warning('iris_model.pkl not found')
            st.toast(':red[Model not found. Run train.py]', icon="🔥")

if __name__ == "__main__":
    run()
