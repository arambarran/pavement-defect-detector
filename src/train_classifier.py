from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.config import CLASSES, IMAGE_EXTENSIONS, MODEL_PATH, PROCESSED_DATASET_DIR
from src.features import extract_hog_features


TRAIN_DIR: Path = PROCESSED_DATASET_DIR / "train"
VAL_DIR: Path = PROCESSED_DATASET_DIR / "val"


def image_paths(class_dir: Path) -> list[Path]:
    # Keep only image files from one class folder.
    paths: list[Path] = []

    for path in class_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)

    return paths


def load_split(split_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    # Convert one split folder into feature rows and numeric labels.
    features: list[np.ndarray] = []
    labels: list[int] = []

    for label, class_name in enumerate(CLASSES):
        class_dir = split_dir / class_name

        for path in image_paths(class_dir):
            features.append(extract_hog_features(path))
            labels.append(label)

    return np.array(features), np.array(labels)


def train_classifier() -> None:
    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        raise FileNotFoundError("Prepared dataset not found. Run: uv run python -m src.prepare_dataset")

    print("Loading training images")
    train_features, train_labels = load_split(TRAIN_DIR)

    print("Loading validation images")
    val_features, val_labels = load_split(VAL_DIR)

    # StandardScaler normalizes feature values before the SVM sees them.
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LinearSVC(class_weight="balanced", max_iter=10_000)),
        ]
    )

    print("Training classifier")
    model.fit(train_features, train_labels)

    # Validation tells us how the model does on images it did not train on.
    predictions = model.predict(val_features)
    accuracy = accuracy_score(val_labels, predictions)

    print()
    print(f"validation accuracy: {accuracy:.3f}")
    print()
    print(classification_report(val_labels, predictions, target_names=CLASSES, zero_division=0))

    # Save the trained model so later scripts can load it for prediction.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Saved model to: {MODEL_PATH}")


def main() -> None:
    train_classifier()


if __name__ == "__main__":
    main()
