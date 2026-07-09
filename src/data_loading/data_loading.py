"""Data-loading step for the Kaggle Digit Recognizer datasets."""

from pathlib import Path
from typing import Any

import pandas as pd

from data_loading.utils.read_csv import read_csv
from data_loading.utils.split_features_and_labels import split_features_and_labels

ROOT_DIR = Path(__file__).resolve().parents[2]

DigitRecognizerData = tuple[pd.DataFrame, pd.Series, pd.DataFrame]


def resolve_project_path(file_path: str | Path) -> Path:
    """Resolve a file path relative to the project root when needed.

    Args:
        file_path: Absolute path or project-root-relative path.

    Returns:
        Resolved file path.
    """
    path = Path(file_path)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def load_digit_recognizer_data(
    data_loading_config: dict[str, Any],
) -> DigitRecognizerData:
    """Load the Kaggle Digit Recognizer train and test datasets.

    Args:
        data_loading_config: Data-loading configuration section.

    Returns:
        A tuple containing training pixel features, training labels, and testing
        pixel features.
    """
    training_path = resolve_project_path(
        file_path=str(data_loading_config["training_path"]),
    )
    testing_path = resolve_project_path(
        file_path=str(data_loading_config["testing_path"]),
    )
    label_column = str(data_loading_config["label_column"])

    df_train = read_csv(file_path=training_path)
    df_test = read_csv(file_path=testing_path)

    x_train, y_train = split_features_and_labels(
        df=df_train,
        label_column=label_column,
    )

    return x_train, y_train, df_test
