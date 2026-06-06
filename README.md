# Pavement Defect Detector

A pothole and pavement defect detector built with scikit-learn and OpenCV. This project is a hands-on introduction to machine learning and computer vision — covering image preprocessing, feature extraction, and classical ML classification without deep learning frameworks.

## Goal

Learn the fundamentals of ML and CV by building a working defect detector from scratch:

- **Computer vision basics** — load, resize, and preprocess road images with OpenCV
- **Feature engineering** — extract meaningful features (texture, edges, color histograms) from images by hand
- **Classical ML** — train and evaluate scikit-learn classifiers (e.g. SVM, Random Forest) on extracted features
- **Model persistence** — save and load trained models with joblib
- **Interactive demo** — explore predictions through a Streamlit UI

## Stack

| Tool | Purpose |
|---|---|
| `scikit-learn` | Classification models and evaluation |
| `opencv-python` | Image loading and preprocessing |
| `numpy` / `pandas` | Feature arrays and data handling |
| `matplotlib` | Visualization |
| `streamlit` | Interactive prediction UI |
| `joblib` | Model serialization |

## Project Structure

```
pavement-defect-detector/
├── data/
│   ├── raw/          # Original road images
│   └── processed/    # Extracted features / preprocessed data
├── models/           # Saved trained models
├── src/              # Source modules (preprocessing, features, training)
├── main.py           # Entry point
└── pyproject.toml
```

## Getting Started

Install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

## Download Dataset

Create a Kaggle API token from your Kaggle account settings, then place it at:

```text
~/.kaggle/kaggle.json
```

Set the expected file permissions:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

Download the binary pothole classification dataset through the Kaggle API:

```bash
uv run python -m src.download_dataset
```

Inspect the downloaded dataset:

```bash
uv run python -m src.inspect_dataset
```

Prepare train, validation, and test folders:

```bash
uv run python -m src.prepare_dataset
```

Train the first classifier:

```bash
uv run python -m src.train_classifier
```

Evaluate the saved classifier on the test set:

```bash
uv run python -m src.evaluate_classifier
```

By default this downloads:

```text
atulyakumar98/pothole-detection-dataset
```

and stores KaggleHub's cache under:

```text
data/raw/kagglehub/
```

Run the project:

```bash
uv run python main.py
```
