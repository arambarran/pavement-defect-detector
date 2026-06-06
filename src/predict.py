from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

from src.config import CLASSES, MODEL_PATH
from src.features import extract_hog_features


def predict_image(image_path: Path) -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run: uv run python -m src.train_classifier")

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load the same trained pipeline used during evaluation.
    model = joblib.load(MODEL_PATH)

    # scikit-learn expects a batch of rows, so one image becomes one row.
    features = extract_hog_features(image_path)
    features_batch = np.array([features])

    prediction = model.predict(features_batch)[0]
    class_name = CLASSES[prediction]

    print(f"image: {image_path}")
    print(f"prediction: {class_name}")

    # LinearSVC uses decision scores instead of probabilities.
    if hasattr(model, "decision_function"):
        decision_score = model.decision_function(features_batch)[0]
        print(f"decision score: {decision_score:.3f}")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: uv run python -m src.predict <image-path>")
        raise SystemExit(1)

    predict_image(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
