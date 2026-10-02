"""XGBoost regressor settings for cycle-time prediction.

Hyperparameters live here so the training pipeline stays an orchestrator.
The random seed is supplied by the caller from project configuration.
"""

from __future__ import annotations

from xgboost import XGBRegressor

XGB_REGRESSOR_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "n_jobs": -1,
    "verbosity": 0,
}


def build_model(random_state: int) -> XGBRegressor:
    """Return an unfitted XGBoost regressor using the parameters above."""
    return XGBRegressor(random_state=random_state, **XGB_REGRESSOR_PARAMS)
