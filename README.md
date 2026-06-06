# Pavement Defect Detector

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![uv](https://img.shields.io/badge/package%20manager-uv-green)
![OpenCV](https://img.shields.io/badge/computer%20vision-OpenCV-red)
![scikit--learn](https://img.shields.io/badge/model-scikit--learn-orange)

Classical computer vision pipeline for classifying road images as `normal` or `potholes`.

This project is built as a learning path for:

- loading datasets from Kaggle
- inspecting and preparing image data
- extracting HOG features with OpenCV
- training a baseline scikit-learn classifier
- evaluating and running single-image predictions

## Stack

| Tool | Purpose |
| --- | --- |
| `uv` | environment and command runner |
| `kagglehub` | Kaggle dataset download |
| `opencv-python` | image loading, resizing, grayscale conversion, HOG features |
| `scikit-learn` | classifier, metrics, model pipeline |
| `numpy` | feature arrays and labels |
| `joblib` | saving and loading trained models |

## Project Layout

```text
pavement-defect-detector/
├── data/                 # ignored raw and processed datasets
├── models/               # ignored trained model artifacts
├── outputs/              # ignored reports and analysis outputs
├── src/
│   ├── config.py
│   ├── download_dataset.py
│   ├── inspect_dataset.py
│   ├── prepare_dataset.py
│   ├── features.py
│   ├── train_classifier.py
│   ├── evaluate_classifier.py
│   ├── analyze_errors.py
│   └── predict.py
├── main.py
├── pyproject.toml
└── README.md
```

## Setup

Install dependencies:

```bash
uv sync
```

## Dataset

Default Kaggle dataset:

```text
atulyakumar98/pothole-detection-dataset
```

KaggleHub stores downloaded files under:

```text
data/raw/kagglehub/
```

Download the dataset:

```bash
uv run python -m src.download_dataset
```

The raw dataset is expected to contain:

```text
normal/
potholes/
```

## Pipeline

Run the full classification workflow one step at a time:

```bash
uv run python -m src.download_dataset
uv run python -m src.inspect_dataset
uv run python -m src.prepare_dataset
uv run python -m src.train_classifier
uv run python -m src.evaluate_classifier
uv run python -m src.analyze_errors
```

Predict one image:

```bash
uv run python -m src.predict path/to/image.jpg
```

## Current Baseline

Model:

```text
HOG features + StandardScaler + LinearSVC
```

Latest observed validation accuracy:

```text
0.891
```

Latest observed test accuracy:

```text
0.856
```

Test confusion matrix:

```text
[[45  9]
 [ 6 44]]
```

Using `potholes` as the positive class:

| Metric | Count | Meaning |
| --- | ---: | --- |
| TN | 45 | actual normal, predicted normal |
| FP | 9 | actual normal, predicted potholes |
| FN | 6 | actual potholes, predicted normal |
| TP | 44 | actual potholes, predicted potholes |

Error analysis report:

```text
outputs/error_analysis.csv
```

## Generated Files

These are created locally and ignored by git:

```text
data/
models/
outputs/
```
