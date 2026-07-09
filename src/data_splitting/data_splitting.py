"""Data-splitting orchestration utilities."""

from typing import Any, TypedDict

import numpy as np

from data_splitting.stratified_train_validation_split import (
    stratified_train_validation_split,
)


class DataSplittingOutput(TypedDict):
    """Output values from the data-splitting step."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray


def split_digit_recognizer_training_data(
    x: np.ndarray,
    y: np.ndarray,
    data_splitting_config: dict[str, Any],
) -> DataSplittingOutput:
    """Split Digit Recognizer labeled data into train and validation sets.

    Args:
        x: Full labeled training feature matrix.
        y: Full labeled training label array.
        data_splitting_config: Data-splitting configuration section.

    Returns:
        Dictionary containing training and validation splits.

    Raises:
        ValueError: If data splitting is disabled.
        ValueError: If the configured split strategy is unsupported.
    """
    if not bool(data_splitting_config["enabled"]):
        raise ValueError("Data splitting must be enabled for the current pipeline.")

    strategy = str(data_splitting_config["strategy"])

    if strategy != "stratified":
        raise ValueError(f"Unsupported data-splitting strategy: {strategy}")

    x_train, y_train, x_validation, y_validation = stratified_train_validation_split(
        x=x,
        y=y,
        validation_size=float(data_splitting_config["validation_size"]),
        random_seed=int(data_splitting_config["random_seed"]),
    )

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_validation": x_validation,
        "y_validation": y_validation,
    }
