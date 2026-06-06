from __future__ import annotations

from pathlib import Path

import cv2

from src.config import CLASSES, IMAGE_EXTENSIONS, RAW_DATASET_DIR


def size_count(size_info: tuple[tuple[int, int], int]) -> int:
    return size_info[1]


def inspect_dataset(dataset_dir: Path = RAW_DATASET_DIR) -> None:
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_dir}. Run: uv run python -m src.download_dataset"
        )

    total_images: int = 0
    unreadable_images: int = 0
    image_sizes: dict[tuple[int, int], int] = {}

    print(f"Dataset: {dataset_dir}")
    print()

    for class_name in CLASSES:
        class_dir = dataset_dir / class_name
        image_paths: list[Path] = []

        for path in class_dir.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(path)

        total_images += len(image_paths)
        print(f"{class_name}: {len(image_paths)} images")

        for path in image_paths:
            image = cv2.imread(str(path))

            if image is None:
                unreadable_images += 1
                continue

            height, width = image.shape[:2]
            image_sizes[(width, height)] = image_sizes.get((width, height), 0) + 1

    print()
    print(f"total: {total_images} images")
    print(f"unreadable: {unreadable_images} images")
    print(f"unique sizes: {len(image_sizes)}")

    print()
    print("most common sizes:")

    for (width, height), count in sorted(image_sizes.items(), key=size_count, reverse=True)[:10]:
        print(f"{width}x{height}: {count}")


def main() -> None:
    inspect_dataset()


if __name__ == "__main__":
    main()
