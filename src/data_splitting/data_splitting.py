"""Data-splitting orchestration utilities."""

from typing import TypedDict

import numpy as np

from data_splitting.stratified_train_validation_split import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_VALIDATION_SIZE,
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
    validation_size: float = DEFAULT_VALIDATION_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> DataSplittingOutput:
    """Split Digit Recognizer labeled data into train and validation sets.

    Args:
        x: Full labeled training feature matrix.
        y: Full labeled training label array.
        validation_size: Fraction of each class assigned to validation.
        random_seed: Seed used to make the split reproducible.

    Returns:
        Dictionary containing training and validation splits.
    """
    x_train, y_train, x_validation, y_validation = stratified_train_validation_split(
        x=x,
        y=y,
        validation_size=validation_size,
        random_seed=random_seed,
    )

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_validation": x_validation,
        "y_validation": y_validation,
    }
