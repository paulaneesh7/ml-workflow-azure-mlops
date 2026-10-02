"""Regression metrics for cycle-time predictions. Values are in days."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error as _mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score


def mean_absolute_error(y_true, y_pred) -> float:
    """Mean absolute error between actual and predicted cycle time."""
    return float(_mean_absolute_error(y_true, y_pred))


def root_mean_squared_error(y_true, y_pred) -> float:
    """Root mean squared error between actual and predicted cycle time."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r_squared(y_true, y_pred) -> float:
    """Coefficient of determination."""
    return float(r2_score(y_true, y_pred))


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return MAE, RMSE, and R² together."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r_squared(y_true, y_pred),
    }
