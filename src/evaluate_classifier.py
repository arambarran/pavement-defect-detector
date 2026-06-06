from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import CLASSES, MODEL_PATH, PROCESSED_DATASET_DIR
from src.dataset import load_feature_split


TEST_DIR: Path = PROCESSED_DATASET_DIR / "test"


def evaluate_classifier() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run: uv run python -m src.train_classifier")

    if not TEST_DIR.exists():
        raise FileNotFoundError("Prepared test dataset not found. Run: uv run python -m src.prepare_dataset")

    print("Loading model")
    model = joblib.load(MODEL_PATH)

    print("Loading test images")
    test_features, test_labels = load_feature_split(TEST_DIR)

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
