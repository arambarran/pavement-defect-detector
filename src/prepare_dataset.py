from __future__ import annotations

import random
import shutil
from pathlib import Path

from src.config import CLASSES, PROCESSED_DATASET_DIR, RAW_DATASET_DIR
from src.dataset import image_paths

SPLITS: tuple[str, ...] = ("train", "val", "test")
RANDOM_SEED: int = 42
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15


def copy_images(paths: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        shutil.copy2(path, output_dir / path.name)


def prepare_dataset() -> None:
    if not RAW_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_DATASET_DIR}. Run: uv run python -m src.download_dataset"
        )

    if PROCESSED_DATASET_DIR.exists():
        shutil.rmtree(PROCESSED_DATASET_DIR)

    print(f"Raw dataset: {RAW_DATASET_DIR}")
    print(f"Processed dataset: {PROCESSED_DATASET_DIR}")
    print()

    for class_name in CLASSES:
        class_dir = RAW_DATASET_DIR / class_name
        paths = image_paths(class_dir)
        random.Random(RANDOM_SEED).shuffle(paths)

        train_count = int(len(paths) * TRAIN_RATIO)
        val_count = int(len(paths) * VAL_RATIO)

        train_paths = paths[:train_count]
        val_paths = paths[train_count : train_count + val_count]
        test_paths = paths[train_count + val_count :]

        split_paths: dict[str, list[Path]] = {
            "train": train_paths,
            "val": val_paths,
            "test": test_paths,
        }

        for split_name in SPLITS:
            output_dir = PROCESSED_DATASET_DIR / split_name / class_name
            copy_images(split_paths[split_name], output_dir)
            print(f"{split_name}/{class_name}: {len(split_paths[split_name])} images")


def main() -> None:
    prepare_dataset()


if __name__ == "__main__":
    main()
