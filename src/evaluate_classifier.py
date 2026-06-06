from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import CLASSES, IMAGE_EXTENSIONS, MODEL_PATH, PROCESSED_DATASET_DIR
from src.features import extract_hog_features


TEST_DIR: Path = PROCESSED_DATASET_DIR / "test"


def image_paths(class_dir: Path) -> list[Path]:
    # Keep only image files from one class folder.
    paths: list[Path] = []

    for path in class_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)

    return paths


def load_test_data() -> tuple[np.ndarray, np.ndarray]:
    # Convert test images into feature rows and numeric labels.
    features: list[np.ndarray] = []
    labels: list[int] = []

    for label, class_name in enumerate(CLASSES):
        class_dir = TEST_DIR / class_name

        for path in image_paths(class_dir):
            features.append(extract_hog_features(path))
            labels.append(label)

    return np.array(features), np.array(labels)


def evaluate_classifier() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run: uv run python -m src.train_classifier")

    if not TEST_DIR.exists():
        raise FileNotFoundError("Prepared test dataset not found. Run: uv run python -m src.prepare_dataset")

    print("Loading model")
    model = joblib.load(MODEL_PATH)

    print("Loading test images")
    test_features, test_labels = load_test_data()

    # The test set gives the most honest score for this trained baseline.
    predictions = model.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)

    print()
    print(f"test accuracy: {accuracy:.3f}")
    print()
    print(classification_report(test_labels, predictions, target_names=CLASSES, zero_division=0))

    print("confusion matrix:")
    print(confusion_matrix(test_labels, predictions))


def main() -> None:
    evaluate_classifier()


if __name__ == "__main__":
    main()
