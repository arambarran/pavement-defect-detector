from __future__ import annotations

import csv
from pathlib import Path

import joblib

from src.config import CLASSES, MODEL_PATH, OUTPUTS_DIR, PROCESSED_DATASET_DIR
from src.dataset import load_feature_split_with_paths


TEST_DIR: Path = PROCESSED_DATASET_DIR / "test"
REPORT_PATH: Path = OUTPUTS_DIR / "error_analysis.csv"


def error_type(actual_label: int, predicted_label: int) -> str:
    if actual_label == predicted_label:
        return "correct"

    if CLASSES[actual_label] == "normal" and CLASSES[predicted_label] == "potholes":
        return "false_positive"

    return "false_negative"


def analyze_errors() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run: uv run python -m src.train_classifier")

    if not TEST_DIR.exists():
        raise FileNotFoundError("Prepared test dataset not found. Run: uv run python -m src.prepare_dataset")

    print("Loading model")
    model = joblib.load(MODEL_PATH)

    print("Loading test images")
    test_features, test_labels, image_paths = load_feature_split_with_paths(TEST_DIR)

    predictions = model.predict(test_features)
    decision_scores = model.decision_function(test_features)

    false_positives: int = 0
    false_negatives: int = 0

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", newline="") as report_file:
        writer = csv.writer(report_file)
        writer.writerow(["image_path", "actual_label", "predicted_label", "decision_score", "error_type"])

        for image_path, actual_label, predicted_label, decision_score in zip(
            image_paths,
            test_labels,
            predictions,
            decision_scores,
            strict=True,
        ):
            current_error_type = error_type(actual_label, predicted_label)

            if current_error_type == "false_positive":
                false_positives += 1

            if current_error_type == "false_negative":
                false_negatives += 1

            writer.writerow(
                [
                    image_path,
                    CLASSES[actual_label],
                    CLASSES[predicted_label],
                    f"{decision_score:.6f}",
                    current_error_type,
                ]
            )

    print()
    print(f"false positives: {false_positives}")
    print(f"false negatives: {false_negatives}")
    print(f"saved report to: {REPORT_PATH}")


def main() -> None:
    analyze_errors()


if __name__ == "__main__":
    main()
