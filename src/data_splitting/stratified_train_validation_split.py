"""Stratified train-validation split utilities."""

from typing import cast

import numpy as np

DEFAULT_VALIDATION_SIZE = 0.2
DEFAULT_RANDOM_SEED = 42

TrainValidationSplit = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def stratified_train_validation_split(
    x: np.ndarray,
    y: np.ndarray,
    validation_size: float = DEFAULT_VALIDATION_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> TrainValidationSplit:
    """Split labeled data into stratified training and validation sets.

    Args:
        x: Feature matrix.
        y: Label array.
        validation_size: Fraction of each class assigned to validation.
        random_seed: Seed used to make the split reproducible.

    Returns:
        Tuple containing training features, training labels, validation features,
        and validation labels.

    Raises:
        ValueError: If x and y have different numbers of examples.
        ValueError: If validation_size is not between 0 and 1.
        ValueError: If a class has fewer than two examples.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must contain the same number of examples.")

    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be greater than 0 and less than 1.")

    rng = np.random.default_rng(seed=random_seed)

    train_index_groups: list[np.ndarray] = []
    validation_index_groups: list[np.ndarray] = []

    for label in np.unique(y):
        class_indices = np.where(y == label)[0]

        if class_indices.size < 2:
            raise ValueError("Each class must contain at least two examples.")

        shuffled_class_indices = class_indices.copy()
        rng.shuffle(shuffled_class_indices)

        validation_count = int(round(shuffled_class_indices.size * validation_size))
        validation_count = max(1, min(validation_count, shuffled_class_indices.size - 1))

        validation_index_groups.append(shuffled_class_indices[:validation_count])
        train_index_groups.append(shuffled_class_indices[validation_count:])

    train_indices = np.concatenate(train_index_groups)
    validation_indices = np.concatenate(validation_index_groups)

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)

    return (
        cast("np.ndarray", x[train_indices]),
        cast("np.ndarray", y[train_indices]),
        cast("np.ndarray", x[validation_indices]),
        cast("np.ndarray", y[validation_indices]),
    )
