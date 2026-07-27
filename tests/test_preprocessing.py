"""Unit tests for the preprocessing functions in src/data_preprocessing.py."""

import pandas as pd
import pytest

from src.data_preprocessing import (
    validate_columns,
    handle_missing_values,
    remove_duplicates,
    split_features_target,
)


# A small handmade dataframe we control completely, so we know exactly
# what every function should do to it. Tests never use the real data here.
def sample_df():
    return pd.DataFrame({
        "age": [52, 40, 52, None],
        "chol": [212, 250, 212, 300],
        "target": [1, 0, 1, 0],
    })


def test_handle_missing_values_fills_nulls():
    """After filling, there should be no missing values left."""
    df = sample_df()
    result = handle_missing_values(df)
    assert result.isnull().sum().sum() == 0


def test_handle_missing_values_does_not_modify_original():
    """The function must work on a copy and leave the caller's df intact."""
    df = sample_df()
    original_null_count = df.isnull().sum().sum()
    handle_missing_values(df)
    # the original still has its missing value; it was not mutated
    assert df.isnull().sum().sum() == original_null_count


def test_remove_duplicates_drops_repeated_rows():
    """Our sample has one duplicate row (age 52); it should be removed."""
    df = sample_df().dropna()  # drop the None row so dup logic is clean
    result = remove_duplicates(df)
    assert len(result) < len(df)


def test_remove_duplicates_returns_new_dataframe():
    """The original dataframe should be unchanged after deduplication."""
    df = sample_df()
    original_length = len(df)
    remove_duplicates(df)
    assert len(df) == original_length


def test_split_features_target_separates_correctly():
    """X must drop the target column; y must be the target column."""
    df = sample_df()
    X, y = split_features_target(df, "target")
    assert "target" not in X.columns
    assert y.name == "target"


def test_validate_columns_raises_on_missing_column():
    """Asking for a column that does not exist must raise ValueError."""
    df = sample_df()
    with pytest.raises(ValueError):
        validate_columns(df, ["age", "does_not_exist"])