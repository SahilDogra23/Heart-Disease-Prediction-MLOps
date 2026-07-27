"""Model validation tests: train on the sample and sanity-check the model."""

import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.data_preprocessing import (
    load_data,
    remove_duplicates,
    split_features_target,
    build_preprocessor,
)

SAMPLE_PATH = "tests/sample_heart.csv"
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


@pytest.fixture
def trained_model():
    """Train a small pipeline and return it with its test set."""
    df = remove_duplicates(load_data(SAMPLE_PATH))
    X, y = split_features_target(df, "target")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    pipeline = Pipeline([
        ("preprocess", build_preprocessor(NUMERIC, CATEGORICAL)),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline, X_test, y_test


def test_predictions_have_correct_shape(trained_model):
    pipeline, X_test, y_test = trained_model
    preds = pipeline.predict(X_test)
    assert len(preds) == len(X_test)


def test_predictions_are_valid_labels(trained_model):
    pipeline, X_test, y_test = trained_model
    preds = pipeline.predict(X_test)
    assert set(np.unique(preds)).issubset({0, 1})


def test_model_meets_minimum_accuracy(trained_model):
    pipeline, X_test, y_test = trained_model
    preds = pipeline.predict(X_test)
    assert accuracy_score(y_test, preds) >= 0.6