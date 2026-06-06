# Pavement Defect Detector

A computer vision pipeline that classifies road images as **normal** or **pothole-damaged** — built entirely with classical CV, no neural networks required.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-HOG%20features-informational)
![scikit-learn](https://img.shields.io/badge/scikit--learn-LinearSVC-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-demo%20app-red)

---

## Demo

Run the interactive app locally:

```bash
uv sync
uv run python -m src.train_classifier   # train once
uv run streamlit run src/app.py         # open the app
```

Upload any road photo and get an instant verdict with model confidence.

> Add screenshots of the app here once running.

---

## How It Works

```
Raw image
    ↓  resize to 128×128
    ↓  HOG (gradient/shape features)  +  HSV color histograms
    ↓  StandardScaler
    ↓  LinearSVC
  Normal / Pothole
```

**Feature extraction** combines two complementary signals:
- **HOG** (Histogram of Oriented Gradients) — captures edge and shape patterns like cracks and surface breaks
- **HSV color histograms** — captures brightness and saturation, since potholes tend to be darker and less saturated than intact road

**Classifier**: a linear SVM with balanced class weights, trained on ~400 labelled images from the [Kaggle pothole detection dataset](https://www.kaggle.com/datasets/atulyakumar98/pothole-detection-dataset).

---

## Results

| Split | Accuracy |
|---|---|
| Validation | **95.0%** |
| Test | **86.5%** |

Test set confusion matrix (50 pothole / 54 normal images):

```
              Predicted
              Normal  Pothole
Actual Normal   44      10
Actual Pothole   4      46
```

| Metric | Value |
|---|---|
| Precision (potholes) | 0.82 |
| Recall (potholes) | 0.92 |
| F1 (potholes) | 0.87 |

The model is tuned to favour recall on potholes — it is better to flag a good road for inspection than to miss real damage.

---

## Stack

| Tool | Role |
|---|---|
| `opencv-python` | image loading, resizing, HOG and HSV feature extraction |
| `scikit-learn` | SVM classifier, pipeline, evaluation metrics |
| `numpy` | feature arrays |
| `joblib` | model serialisation |
| `streamlit` | interactive demo app |
| `kagglehub` | dataset download |
| `uv` | dependency management and task runner |

---

## Project Layout

```
pavement-defect-detector/
├── src/
│   ├── config.py              # paths and constants
│   ├── download_dataset.py    # fetch from Kaggle
│   ├── prepare_dataset.py     # train/val/test split
│   ├── features.py            # HOG + color histogram extraction
│   ├── train_classifier.py    # fit and save the model
│   ├── evaluate_classifier.py # test-set metrics
│   ├── analyze_errors.py      # per-image error report
│   ├── visualize_errors.py    # false positive/negative grids
│   ├── predict.py             # single-image CLI prediction
│   └── app.py                 # Streamlit demo
├── data/                      # raw and processed datasets (git-ignored)
├── models/                    # saved model artifacts (git-ignored)
├── outputs/                   # reports and visualisations (git-ignored)
└── pyproject.toml
```

---

## Pipeline

```bash
uv run python -m src.download_dataset    # download Kaggle dataset
uv run python -m src.prepare_dataset    # split into train/val/test
uv run python -m src.train_classifier   # fit the model
uv run python -m src.evaluate_classifier
uv run python -m src.analyze_errors     # CSV report of all errors
uv run python -m src.visualize_errors   # image grids of FP/FN
uv run python -m src.predict path/to/image.jpg
```
