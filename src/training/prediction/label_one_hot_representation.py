"""Create one-hot representations of categorical ground-truth labels."""

from typing import cast

import numpy as np

NUM_CATEGORIES = 10


def label_one_hot_representation(labels: np.ndarray) -> np.ndarray:
    """Represent categorical ground-truth labels as a one-hot matrix.

    Args:
        labels: Ground-truth class labels with shape examples.

    Returns:
        Dense one-hot representation of the labels.
    """
    one_hot = np.eye(NUM_CATEGORIES)[labels]

    return cast("np.ndarray", one_hot)
