from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.config import IMAGE_SIZE


# HOG turns an image into edge/shape features that scikit-learn can train on.
HOG_DESCRIPTOR = cv2.HOGDescriptor(
    IMAGE_SIZE,
    (16, 16),
    (8, 8),
    (8, 8),
    9,
)


def load_image(path: Path) -> np.ndarray:
    # OpenCV returns None when a file is missing or unreadable.
    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    return image


def preprocess_image(image: np.ndarray) -> np.ndarray:
    # Every image needs the same size and color format before feature extraction.
    resized_image = cv2.resize(image, IMAGE_SIZE)
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    return grayscale_image


def extract_hog_features_from_image(image: np.ndarray) -> np.ndarray:
    # This is the full image-to-numbers step used by the classifier.
    processed_image = preprocess_image(image)
    features = HOG_DESCRIPTOR.compute(processed_image)

    return features.flatten()


def extract_hog_features(path: Path) -> np.ndarray:
    image = load_image(path)

    return extract_hog_features_from_image(image)
