"""Utilities for reading CSV files."""

from pathlib import Path

import pandas as pd


def read_csv(file_path: str | Path) -> pd.DataFrame:
    """Read a CSV file into a pandas DataFrame.

    Args:
        file_path: Path to the CSV file.

    Returns:
        Loaded CSV data as a pandas DataFrame.
    """
    return pd.read_csv(filepath_or_buffer=file_path)
