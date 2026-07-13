"""Batching utilities for training loops."""

from collections.abc import Iterator
from typing import Any, cast

import numpy as np

SUPPORTED_BATCHING_STRATEGIES = {
    "full_batch",
    "mini_batch",
}


def get_batching_config(
    training_config: dict[str, Any],
) -> dict[str, Any]:
    """Get batching configuration with backward-compatible defaults.

    Args:
        training_config: Training configuration section.

    Returns:
        Batching configuration dictionary.
    """
    batching_config = training_config.get("batching")

    if not isinstance(batching_config, dict):
        return {
            "strategy": "full_batch",
            "batch_size": None,
            "shuffle": False,
            "random_seed": 42,
        }

    return cast(dict[str, Any], batching_config)


def iter_mini_batches(
    x_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int,
    shuffle: bool,
    random_generator: np.random.Generator,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield mini-batches from the training data.

    Args:
        x_train: Training feature matrix.
        y_train: Training label array.
        batch_size: Number of examples per mini-batch.
        shuffle: Whether to shuffle examples before batching.
        random_generator: NumPy random generator used for deterministic shuffling.

    Yields:
        Tuples containing mini-batch features and labels.
    """
    sample_count = x_train.shape[0]
    indices = np.arange(sample_count)

    if shuffle:
        random_generator.shuffle(indices)

    for batch_start in range(0, sample_count, batch_size):
        batch_end = batch_start + batch_size
        batch_indices = indices[batch_start:batch_end]

        yield x_train[batch_indices], y_train[batch_indices]
