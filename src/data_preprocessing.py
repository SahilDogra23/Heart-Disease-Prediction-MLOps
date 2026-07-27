"""Data loading and preprocessing for the heart disease dataset.

Each function does one job and returns a NEW dataframe rather than
modifying its input. That property is what makes them testable later.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_data(path):
    """Read the raw CSV into a dataframe."""
    return pd.read_csv(path)


def validate_columns(df, required_columns):
    """Raise a clear error if any expected column is missing."""
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True


def handle_missing_values(df):
    """Return a copy with missing values filled.

    Numeric columns get the median, others get the mode. Uses a copy so
    the caller's original dataframe is never touched.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    return df


def remove_duplicates(df):
    """Return a copy with duplicate rows dropped (your notebook found 723)."""
    return df.drop_duplicates().reset_index(drop=True)


def split_features_target(df, target_column):
    """Separate the label column from the feature columns."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def build_preprocessor(numeric_features, categorical_features):
    """Build a scikit-learn transformer that scales numeric features and
    one-hot encodes categorical ones. Returned unfitted so training and
    testing can each fit it on their own data.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )