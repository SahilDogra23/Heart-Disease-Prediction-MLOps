"""Data validation tests: verify properties of the dataset sample."""

import pytest

from src.data_preprocessing import load_data

SAMPLE_PATH = "tests/sample_heart.csv"

EXPECTED_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]


@pytest.fixture
def df():
    """Load the sample once and hand it to any test that takes a 'df' argument."""
    return load_data(SAMPLE_PATH)


def test_expected_columns_present(df):
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    assert not missing, f"Missing columns: {missing}"


def test_target_is_binary(df):
    assert set(df["target"].unique()).issubset({0, 1})


def test_age_within_reasonable_range(df):
    assert df["age"].min() >= 0
    assert df["age"].max() <= 120


def test_cholesterol_non_negative(df):
    assert (df["chol"] >= 0).all()