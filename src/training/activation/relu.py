"""ReLU activation utilities."""

from typing import cast

import numpy as np


def relu(pre_activation: np.ndarray) -> np.ndarray:
    """Apply the ReLU activation function element-wise.

    Args:
        pre_activation: NumPy array containing pre-activation values.

    Returns:
        NumPy array with negative values replaced by zero and positive values
        preserved.
    """
    return cast(np.ndarray, np.maximum(0, pre_activation))
