"""Row-level feature preparation before the train/validation/test split.

Transformations here do not learn statistics from the dataset. Median
imputation and categorical encoding are fit later, on the training split only.
"""

from __future__ import annotations

import pandas as pd

# Planning dates known when the request is opened. They are expanded into
# numeric calendar features and then dropped so raw timestamps are not encoded
# as high-cardinality categories.
_DATE_COLUMNS = ("created_at", "requested_implementation_date")

_NUMERIC_COLUMNS = (
    "required_approver_count",
    "assignment_backlog_size",
    "assignee_experience_months",
    "dependency_count",
    "affected_system_count",
    "estimated_effort_hours",
    "planned_downtime_minutes",
    "description_length",
    "created_hour",
    "created_day_of_week",
    "created_month",
    "requested_lead_time_days",
)

_MISSING_CATEGORY = "Unknown"


def engineer_features(features: pd.DataFrame) -> pd.DataFrame:
    """Build the candidate feature table from pre-resolution columns.

    Expects leakage columns and the target to already be removed. Blank
    strings become missing values. Categorical gaps are filled with a fixed
    token rather than a mode learned from the data. Numeric gaps stay missing
    so a train-fitted imputer can fill them.
    """
    frame = features.copy()
    frame = _replace_blank_strings(frame)
    frame = _add_calendar_features(frame)
    frame = _coerce_numeric_columns(frame)
    return _fill_categorical_missing(frame)


def _replace_blank_strings(features: pd.DataFrame) -> pd.DataFrame:
    object_columns = features.select_dtypes(include=["object", "string"]).columns
    if len(object_columns) == 0:
        return features
    features = features.copy()
    features[object_columns] = features[object_columns].replace(r"^\s*$", pd.NA, regex=True)
    return features


def _add_calendar_features(features: pd.DataFrame) -> pd.DataFrame:
    """Derive schedule features that are known before the change is resolved."""
    features = features.copy()
    created_at = _parse_datetime(features, "created_at")
    requested_on = _parse_datetime(features, "requested_implementation_date")

    if created_at is not None:
        features["created_hour"] = created_at.dt.hour
        features["created_day_of_week"] = created_at.dt.dayofweek
        features["created_month"] = created_at.dt.month
    if created_at is not None and requested_on is not None:
        lead_time = requested_on - created_at
        features["requested_lead_time_days"] = lead_time.dt.total_seconds() / 86400

    columns_to_drop = [column for column in _DATE_COLUMNS if column in features.columns]
    return features.drop(columns=columns_to_drop)


def _parse_datetime(features: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in features.columns:
        return None
    return pd.to_datetime(features[column], errors="coerce")


def _coerce_numeric_columns(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    for column in _NUMERIC_COLUMNS:
        if column in features.columns:
            features[column] = pd.to_numeric(features[column], errors="coerce")
    return features


def _fill_categorical_missing(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    categorical_columns = features.select_dtypes(include=["object", "string", "category"]).columns
    for column in categorical_columns:
        features[column] = features[column].fillna(_MISSING_CATEGORY).astype("object")
    return features
