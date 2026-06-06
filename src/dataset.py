from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import CLASSES, IMAGE_EXTENSIONS
from src.features import extract_hog_features


def image_paths(class_dir: Path) -> list[Path]:
    paths: list[Path] = []

    for path in class_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)

    return paths


def load_feature_split(split_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    # Convert a train, validation, or test folder into features and labels.
    features: list[np.ndarray] = []
    labels: list[int] = []

    for label, class_name in enumerate(CLASSES):
        class_dir = split_dir / class_name

        for path in image_paths(class_dir):
            features.append(extract_hog_features(path))
            labels.append(label)

    return np.array(features), np.array(labels)


def load_feature_split_with_paths(split_dir: Path) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    # Keep paths when we need to inspect which specific images were wrong.
    features: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[Path] = []

    for label, class_name in enumerate(CLASSES):
        class_dir = split_dir / class_name

        for path in image_paths(class_dir):
            features.append(extract_hog_features(path))
            labels.append(label)
            paths.append(path)

    return np.array(features), np.array(labels), paths
