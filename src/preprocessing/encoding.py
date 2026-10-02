"""Fit categorical and numeric preprocessing on the training split only.

The returned scikit-learn transformer is the inference artifact: transform
validation, test, and future batches with the same fitted object. Do not
refit it on those batches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def fit_encoder(X_train: pd.DataFrame) -> ColumnTransformer:
    """Learn imputation and one-hot categories from the training features."""
    numeric_columns, categorical_columns = _column_groups(X_train)
    if not numeric_columns and not categorical_columns:
        raise ValueError("No feature columns are available to encode.")

    transformers = []
    if numeric_columns:
        # Medians are estimated from training rows only.
        transformers.append(
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_columns,
            )
        )
    if categorical_columns:
        # most_frequent is a fallback for any categorical value still missing
        # after feature engineering. Unknown categories at inference are ignored.
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "one_hot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    encoder = ColumnTransformer(transformers=transformers, remainder="drop")
    encoder.fit(X_train)
    return encoder


def transform_features(
    encoder: ColumnTransformer,
    features: pd.DataFrame,
) -> np.ndarray:
    """Apply a fitted encoder. Column order in ``features`` does not matter."""
    transformed = encoder.transform(features)
    return np.asarray(transformed)


def _column_groups(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_columns = [
        column
        for column in features.columns
        if pd.api.types.is_numeric_dtype(features[column])
    ]
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]
    return numeric_columns, categorical_columns
