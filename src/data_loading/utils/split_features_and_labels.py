"""Utilities for separating input features from target labels."""

import pandas as pd

DEFAULT_LABEL_COLUMN = "label"


def split_features_and_labels(
    df: pd.DataFrame,
    label_column: str = DEFAULT_LABEL_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate a labeled DataFrame into feature columns and target labels.

    Args:
        df: DataFrame containing feature columns and one label column.
        label_column: Name of the target label column.

    Returns:
        A tuple containing the feature DataFrame and label Series.

    Raises:
        ValueError: If the label column is missing.
    """
    if label_column not in df.columns:
        raise ValueError(
            f"Data Integrity Failure: expected label column '{label_column}' was not found."
        )

    features = df.drop(columns=[label_column])
    labels = df[label_column]

    return features, labels
