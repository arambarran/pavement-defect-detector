from __future__ import annotations

import os
from pathlib import Path

import kagglehub


DATASET: str = "atulyakumar98/pothole-detection-dataset"
CACHE_DIR: Path = Path("data/raw/kagglehub")


def download_dataset(dataset: str = DATASET, cache_dir: Path = CACHE_DIR) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = str(cache_dir)

    downloaded_path = kagglehub.dataset_download(dataset)
    return Path(downloaded_path)


def main() -> None:
    dataset_path = download_dataset()
    print(f"Downloaded to: {dataset_path}")


if __name__ == "__main__":
    main()
