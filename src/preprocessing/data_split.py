"""Split features and the target into train, validation, and test sets.

The public function accepts a strategy name so a grouped or time-based split
can replace the random split later without changing the training pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSplit:
    """Hold the three modeling splits. Indexes are reset and not meaningful."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
    strategy: str = "random",
    time_column: str | None = None,
    group_column: str | None = None,
) -> DatasetSplit:
    """Split rows according to ``strategy``.

    ``time_column`` and ``group_column`` are accepted now so a future strategy
    can use them. Only ``strategy="random"`` is implemented.
    """
    _validate_ratios(train_ratio, validation_ratio, test_ratio)
    if len(features) != len(target):
        raise ValueError("Features and target must have the same number of rows.")

    if strategy == "random":
        return _random_split(
            features, target, train_ratio, validation_ratio, test_ratio, seed
        )
    if strategy == "time":
        raise NotImplementedError(
            "Time-based splitting is not implemented yet. Sort by "
            f"{time_column or 'a timestamp column'} and cut contiguous "
            "train, validation, and test windows so later periods are not "
            "used to train earlier ones."
        )
    if strategy == "group":
        raise NotImplementedError(
            "Group-based splitting is not implemented yet. Keep every value of "
            f"{group_column or 'a group column'} inside a single split so "
            "related requests do not leak across train and test."
        )
    raise ValueError(
        f"Unknown split strategy '{strategy}'. Expected 'random', 'time', or 'group'."
    )


def _validate_ratios(train_ratio: float, validation_ratio: float, test_ratio: float) -> None:
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("Train, validation, and test ratios must all be positive.")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(
            "Train, validation, and test ratios must sum to 1. "
            f"Received {train_ratio}, {validation_ratio}, {test_ratio}."
        )


def _random_split(
    features: pd.DataFrame,
    target: pd.Series,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> DatasetSplit:
    holdout_ratio = validation_ratio + test_ratio
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        features,
        target,
        test_size=holdout_ratio,
        random_state=seed,
        shuffle=True,
    )
    test_fraction_of_holdout = test_ratio / holdout_ratio
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=test_fraction_of_holdout,
        random_state=seed,
        shuffle=True,
    )
    return DatasetSplit(
        X_train=_reset(X_train),
        X_validation=_reset(X_validation),
        X_test=_reset(X_test),
        y_train=_reset_series(y_train),
        y_validation=_reset_series(y_validation),
        y_test=_reset_series(y_test),
    )


def _reset(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.reset_index(drop=True)


def _reset_series(series: pd.Series) -> pd.Series:
    return series.reset_index(drop=True)
