from __future__ import annotations

import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import CLASSES, MODEL_PATH  # noqa: E402
from src.features import extract_hog_features_from_image  # noqa: E402


st.set_page_config(page_title="Pavement Defect Detector", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def read_uploaded_image(uploaded_file) -> np.ndarray:
    file_bytes = uploaded_file.getvalue()
    image_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("The uploaded file could not be read as an image.")

    return image


def predict_uploaded_image(image: np.ndarray) -> tuple[str, float]:
    model = load_model()
    features = extract_hog_features_from_image(image)
    features_batch = np.array([features])

    prediction = model.predict(features_batch)[0]
    decision_score = model.decision_function(features_batch)[0]

    return CLASSES[prediction], float(decision_score)


def main() -> None:
    st.title("Pavement Defect Detector")
    st.caption("Classifies a road image as normal or potholes using HOG features and a linear SVM.")

    if not MODEL_PATH.exists():
        st.error("Trained model not found. Run `uv run python -m src.train_classifier` first.")
        return

    uploaded_file = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is None:
        st.info("Upload an image to run a prediction.")
        return

    st.image(uploaded_file, caption="Uploaded image", width="stretch")

    try:
        image = read_uploaded_image(uploaded_file)
        prediction, decision_score = predict_uploaded_image(image)
    except ValueError as error:
        st.error(str(error))
        return

    st.subheader("Prediction")
    st.metric("Class", prediction)
    st.metric("Decision score", f"{decision_score:.3f}")

    if decision_score >= 0:
        st.info("Positive scores lean toward potholes.")
        return

    st.info("Negative scores lean toward normal.")


if __name__ == "__main__":
    main()
