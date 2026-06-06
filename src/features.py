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

COLOR_BINS: int = 32


def load_image(path: Path) -> np.ndarray:
    # OpenCV returns None when a file is missing or unreadable.
    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    return image


def preprocess_image(image: np.ndarray) -> np.ndarray:
    # Every image needs the same size and color format before feature extraction.
    return cv2.resize(image, IMAGE_SIZE)


def extract_hog_features_from_image(image: np.ndarray) -> np.ndarray:
    resized = preprocess_image(image)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hog = HOG_DESCRIPTOR.compute(gray).flatten()

    # HSV color histograms capture brightness and saturation cues that HOG misses.
    # Potholes tend to be darker and more desaturated than intact road surface.
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    color_hists = [
        cv2.calcHist([hsv], [ch], None, [COLOR_BINS], [0, 256]).flatten()
        for ch in range(3)
    ]
    color_features = np.concatenate(color_hists)
    color_features /= color_features.sum() + 1e-6

    return np.concatenate([hog, color_features])


def extract_hog_features(path: Path) -> np.ndarray:
    image = load_image(path)

    return extract_hog_features_from_image(image)
