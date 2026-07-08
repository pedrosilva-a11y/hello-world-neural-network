"""Convert pandas DataFrames to NumPy arrays."""

from typing import cast

import numpy as np
import pandas as pd


def convert_dataframe_to_matrix(df: pd.DataFrame) -> np.ndarray:
    """Convert a pandas DataFrame to a NumPy 2D array.

    Args:
        df: Input DataFrame.

    Returns:
        Input DataFrame converted to a NumPy 2D array.
    """
    return cast("np.ndarray", df.to_numpy())
