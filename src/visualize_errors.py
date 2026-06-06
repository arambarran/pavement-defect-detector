from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from src.analyze_errors import REPORT_PATH
from src.config import OUTPUTS_DIR


FALSE_POSITIVES_PATH: Path = OUTPUTS_DIR / "false_positives.png"
FALSE_NEGATIVES_PATH: Path = OUTPUTS_DIR / "false_negatives.png"
THUMBNAIL_SIZE: tuple[int, int] = (160, 120)
GRID_COLUMNS: int = 3


def error_image_paths(error_name: str) -> list[Path]:
    paths: list[Path] = []

    with REPORT_PATH.open() as report_file:
        reader = csv.DictReader(report_file)

        for row in reader:
            if row["error_type"] == error_name:
                paths.append(Path(row["image_path"]))

    return paths


def make_thumbnail(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    thumbnail = cv2.resize(image, THUMBNAIL_SIZE)
    label = path.parent.name

    cv2.rectangle(thumbnail, (0, 0), (THUMBNAIL_SIZE[0], 24), (0, 0, 0), -1)
    cv2.putText(thumbnail, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return thumbnail


def make_grid(paths: list[Path]) -> np.ndarray:
    rows: list[np.ndarray] = []

    for start in range(0, len(paths), GRID_COLUMNS):
        row_paths = paths[start : start + GRID_COLUMNS]
        thumbnails = [make_thumbnail(path) for path in row_paths]

        while len(thumbnails) < GRID_COLUMNS:
            thumbnails.append(np.zeros((THUMBNAIL_SIZE[1], THUMBNAIL_SIZE[0], 3), dtype=np.uint8))

        rows.append(np.hstack(thumbnails))

    return np.vstack(rows)


def save_error_grid(error_name: str, output_path: Path) -> None:
    paths = error_image_paths(error_name)

    if not paths:
        print(f"no {error_name} images found")
        return

    grid = make_grid(paths)
    cv2.imwrite(str(output_path), grid)

    print(f"saved {error_name}: {output_path}")


def visualize_errors() -> None:
    if not REPORT_PATH.exists():
        raise FileNotFoundError("Error report not found. Run: uv run python -m src.analyze_errors")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    save_error_grid("false_positive", FALSE_POSITIVES_PATH)
    save_error_grid("false_negative", FALSE_NEGATIVES_PATH)


def main() -> None:
    visualize_errors()


if __name__ == "__main__":
    main()
