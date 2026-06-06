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


st.set_page_config(
    page_title="Pavement Defect Detector",
    page_icon="🛣️",
    layout="centered",
)

st.markdown(
    """
    <style>
    .verdict-box {
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 8px;
        margin-bottom: 8px;
    }
    .verdict-normal {
        background-color: #0d3320;
        border: 1px solid #1a6640;
    }
    .verdict-pothole {
        background-color: #3a1a1a;
        border: 1px solid #8b3a3a;
    }
    .verdict-label {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .verdict-sub {
        font-size: 13px;
        opacity: 0.7;
    }
    .section-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        opacity: 0.5;
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def confidence_from_score(decision_score: float) -> float:
    """Map decision score to a 0–1 confidence value via sigmoid."""
    return 1.0 / (1.0 + np.exp(-abs(decision_score)))


def render_result(prediction: str, decision_score: float) -> None:
    confidence = confidence_from_score(decision_score)
    is_pothole = prediction == "potholes"

    icon = "⚠️" if is_pothole else "✅"
    label = "Pothole detected" if is_pothole else "Normal road surface"
    box_class = "verdict-pothole" if is_pothole else "verdict-normal"
    sub = "Road damage identified — inspection recommended." if is_pothole else "No visible defects detected."

    st.markdown(
        f"""
        <div class="verdict-box {box_class}">
            <div class="verdict-label">{icon}&nbsp; {label}</div>
            <div class="verdict-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Model confidence</div>', unsafe_allow_html=True)
    st.progress(confidence, text=f"{confidence * 100:.0f}%")

    with st.expander("Details"):
        st.caption(
            f"Raw decision score: `{decision_score:.4f}` — "
            "positive values lean toward potholes, negative toward normal."
        )
        st.caption(
            "Features: HOG (shape/gradient) + HSV color histograms. "
            "Classifier: LinearSVC trained on the Kaggle pothole detection dataset."
        )


def main() -> None:
    st.title("🛣️ Pavement Defect Detector")
    st.caption(
        "Upload a road photo to instantly classify it as normal or pothole-damaged — "
        "no GPU, no deep learning, just computer vision."
    )

    st.divider()

    if not MODEL_PATH.exists():
        st.error(
            "Trained model not found. Run `uv run python -m src.train_classifier` first.",
            icon="🚫",
        )
        return

    uploaded_file = st.file_uploader(
        "Drop a road image here",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown(
            '<div style="text-align:center; opacity:0.4; padding: 40px 0; font-size:15px;">'
            "📂 Upload a road image to run a prediction"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    img_col, result_col = st.columns([1.1, 1], gap="large")

    with img_col:
        st.markdown('<div class="section-label">Input image</div>', unsafe_allow_html=True)
        st.image(uploaded_file, use_container_width=True)

    with result_col:
        st.markdown('<div class="section-label">Analysis</div>', unsafe_allow_html=True)

        try:
            image = read_uploaded_image(uploaded_file)
            prediction, decision_score = predict_uploaded_image(image)
        except ValueError as error:
            st.error(str(error))
            return

        render_result(prediction, decision_score)


if __name__ == "__main__":
    main()