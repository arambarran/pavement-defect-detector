from __future__ import annotations

import os
from pathlib import Path

import kagglehub

from src.config import DATASET, KAGGLE_CACHE_DIR


def download_dataset(dataset: str = DATASET, cache_dir: Path = KAGGLE_CACHE_DIR) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = str(cache_dir)

    downloaded_path = kagglehub.dataset_download(dataset)
    return Path(downloaded_path)


def main() -> None:
    dataset_path = download_dataset()
    print(f"Downloaded to: {dataset_path}")


if __name__ == "__main__":
    main()
