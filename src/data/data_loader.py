"""Load the change-request CSV supplied at runtime.

The path comes from the caller (locally, or from the Azure ML data asset
mount). This module does not choose a dataset location.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def resolve_csv_path(data_path: str | Path) -> Path:
    """Return a CSV file path from a file or a mounted data-asset directory.

    Azure ML may pass either the CSV itself or a directory that contains it.
    A single CSV in that directory is used. If several are present, pass the
    file path explicitly.
    """
    path = Path(data_path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    csv_files = sorted(path.rglob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found under {path}")
    raise FileNotFoundError(
        f"Multiple CSV files found under {path}. Pass the specific CSV path."
    )


def load_csv(csv_path: str | Path) -> pd.DataFrame:
    """Read a change-request CSV into a DataFrame."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)
