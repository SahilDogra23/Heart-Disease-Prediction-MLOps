# Heart Disease Prediction: MLOps Pipeline

An end-to-end machine learning project that predicts the presence of heart
disease from clinical measurements, built with a full MLOps workflow:
modular code, data versioning, experiment tracking, automated testing,
CI/CD, and drift monitoring.

## Project overview

The model predicts whether a patient has heart disease (binary classification)
from 13 clinical features such as age, resting blood pressure, cholesterol, and
maximum heart rate. **Recall is the primary metric**, because in a medical screen
a false negative (missing a sick patient) is costlier than a false alarm.

After comparing five-plus experiments across logistic regression, random forest,
and SVM, **logistic regression was selected** as the best model:

| Metric | Score |
|---|---|
| Recall | 0.85 |
| Accuracy | 0.84 |
| Precision | 0.85 |
| F1 | 0.85 |

Notably, the simplest model outperformed the more complex ones, which is common
on small tabular datasets.

## Tech stack

- **Data versioning:** DVC
- **Experiment tracking:** MLflow
- **Testing:** pytest
- **CI/CD:** GitHub Actions
- **Drift monitoring:** Evidently
- **Modeling:** scikit-learn, pandas
- **Serving:** FastAPI + Streamlit + Docker

## Project structure

```
.
├── src/
│   ├── data_preprocessing.py   # loading, cleaning, feature prep
│   ├── train.py                # config-driven training + MLflow logging + perf gate
│   ├── evaluate.py             # evaluation metrics
│   └── monitor_drift.py        # Evidently drift detection
├── configs/
│   ├── config.yaml             # main training config
│   └── ci_config.yaml          # lightweight config for CI
├── tests/
│   ├── test_preprocessing.py   # unit tests
│   ├── test_data.py            # data validation tests
│   ├── test_model.py           # model validation tests
│   └── sample_heart.csv        # small sample for CI-safe tests
├── .github/workflows/ci.yml    # CI/CD pipeline
├── compare_experiments.py      # queries MLflow to find the best run
├── MONITORING.md               # drift analysis writeup
├── requirements.txt
└── data/heart.csv.dvc          # DVC pointer (data itself is not in Git)
```

## Setup

```bash
git clone https://github.com/SahilDogra23/Heart-Disease-Prediction-MLOps.git
cd Heart-Disease-Prediction-MLOps
pip install -r requirements.txt
```

### Getting the data

The dataset is versioned with DVC and is not committed to Git. The DVC remote
for this project is local, so if `dvc pull` is unavailable on your machine,
download `heart.csv` from the
[Kaggle Heart Disease dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
and place it at `data/heart.csv`.

```bash
dvc pull   # if the remote is accessible
```

## Running the project

**Train the model** (reads all settings from the config):
```bash
python -m src.train --config configs/config.yaml
```

**View experiments** in the MLflow dashboard:
```bash
mlflow ui
# then open http://127.0.0.1:5000
```

**Find the best run** programmatically:
```bash
python compare_experiments.py
```

**Run the tests:**
```bash
python -m pytest tests/ -v
```

**Check for data drift** (saves an HTML report to `reports/`):
```bash
python -m src.monitor_drift
```

## How it is built

**Data versioning (DVC).** The dataset is tracked with DVC, so Git holds only a
small pointer file while the data lives in a remote. This keeps large files out
of Git while preserving reproducibility.

**Experiment tracking (MLflow).** Every training run logs its parameters,
metrics, and the trained model. `compare_experiments.py` queries all runs to
identify the best by recall.

**Testing (pytest).** Thirteen tests across three levels: unit tests for
preprocessing functions, data validation tests for the dataset, and model
validation tests that train and sanity-check a model.

**CI/CD (GitHub Actions).** On every push, a test job runs the full suite, and a
train job (which runs only if tests pass) trains the model and fails the build if
it misses performance thresholds.

**Drift monitoring (Evidently).** `src/monitor_drift.py` compares the training
distribution against simulated production data, reports drifted features, and
exits non-zero if drift exceeds a threshold. See `MONITORING.md` for the analysis.

