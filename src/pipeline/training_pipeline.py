"""Orchestrate change-request cycle-time training.

Azure ML passes the data-asset path and artifact directory in as arguments.
Workspace, compute, and environment settings stay in ``jobs/train.yml``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import load_csv, resolve_csv_path
from src.model.train_config import build_model
from src.preprocessing.data_split import split_dataset
from src.preprocessing.encoding import fit_encoder, transform_features
from src.preprocessing.feature_engineering import engineer_features
from src.utils.metrics import regression_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost model to predict change-request cycle time."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="CSV file, or a directory containing the change-request CSV.",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the non-secret project configuration.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the model and preprocessing artifacts. "
        "Defaults to model_output_dir in the config file.",
    )
    return parser.parse_args()


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.is_file():
        path = PROJECT_ROOT / config_path
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return config


def prepare_features(frame: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Drop leakage columns, separate the target, and engineer features."""
    target_column = config["target_column"]
    leakage_columns = list(config["leakage_columns"])
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' is missing from the dataset.")

    missing_leakage = [column for column in leakage_columns if column not in frame.columns]
    if missing_leakage:
        raise ValueError(
            "Expected leakage columns are missing from the dataset: "
            + ", ".join(missing_leakage)
        )

    # Closure, resolution, and identifier fields are removed before any
    # statistic is fit so they cannot influence the model.
    features = frame.drop(columns=leakage_columns + [target_column])
    target = frame[target_column].astype(float)
    features = engineer_features(features)

    leaked = set(features.columns) & set(leakage_columns)
    if leaked or target_column in features.columns:
        raise RuntimeError(f"Target or leakage columns remained in features: {leaked}")
    return features, target


def save_artifacts(model, encoder, metrics: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    joblib.dump(encoder, output_dir / "preprocessor.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _print_metrics(split_name: str, metrics: dict[str, float]) -> None:
    print(f"{split_name} MAE: {metrics['mae']:.4f}")
    print(f"{split_name} RMSE: {metrics['rmse']:.4f}")
    print(f"{split_name} R²: {metrics['r2']:.4f}")


def run_training(data_path: str, config_path: str, output_dir: str | None) -> None:
    config = load_config(config_path)
    csv_path = resolve_csv_path(data_path)
    frame = load_csv(csv_path)
    print(f"Experiment: {config['experiment_name']}")
    print(f"Loaded {len(frame)} rows from {csv_path}")

    features, target = prepare_features(frame, config)
    split_config = config["split"]
    splits = split_dataset(
        features,
        target,
        train_ratio=float(split_config["train_ratio"]),
        validation_ratio=float(split_config["validation_ratio"]),
        test_ratio=float(split_config["test_ratio"]),
        seed=int(config["seed"]),
        strategy=str(split_config.get("strategy", "random")),
    )
    print(
        "Split sizes: "
        f"train={len(splits.X_train)}, "
        f"validation={len(splits.X_validation)}, "
        f"test={len(splits.X_test)}"
    )

    # Fit on training rows only, then reuse the encoder for the other splits.
    encoder = fit_encoder(splits.X_train)
    X_train = transform_features(encoder, splits.X_train)
    X_validation = transform_features(encoder, splits.X_validation)
    X_test = transform_features(encoder, splits.X_test)

    model = build_model(random_state=int(config["seed"]))
    model.fit(X_train, splits.y_train)

    validation_metrics = regression_metrics(
        splits.y_validation, model.predict(X_validation)
    )
    test_metrics = regression_metrics(splits.y_test, model.predict(X_test))
    _print_metrics("Validation", validation_metrics)
    _print_metrics("Test", test_metrics)

    artifact_dir = Path(output_dir or config["model_output_dir"])
    if not artifact_dir.is_absolute():
        artifact_dir = PROJECT_ROOT / artifact_dir
    save_artifacts(
        model,
        encoder,
        {"validation": validation_metrics, "test": test_metrics},
        artifact_dir,
    )
    print(f"Saved model and preprocessor to {artifact_dir}")


def main() -> None:
    args = parse_args()
    run_training(args.data_path, args.config, args.output_dir)


if __name__ == "__main__":
    main()
