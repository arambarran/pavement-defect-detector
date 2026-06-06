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

Run the project:

```bash
uv run python main.py
```
