from pathlib import Path


DATASET: str = "atulyakumar98/pothole-detection-dataset"
KAGGLE_CACHE_DIR: Path = Path("data/raw/kagglehub")
RAW_DATASET_DIR: Path = KAGGLE_CACHE_DIR / "datasets/atulyakumar98/pothole-detection-dataset/versions/4"
PROCESSED_DATASET_DIR: Path = Path("data/processed/classification")
MODEL_PATH: Path = Path("models/classifier.joblib")
OUTPUTS_DIR: Path = Path("outputs")
CLASSES: tuple[str, ...] = ("normal", "potholes")
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
IMAGE_SIZE: tuple[int, int] = (128, 128)
