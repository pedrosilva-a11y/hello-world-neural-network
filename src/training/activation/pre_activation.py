"""Defines the linear pre-activation computation for a neural network layer."""

import numpy as np


def compute_pre_activation(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute the linear pre-activation values for a neural network layer.

    Args:
        x: Input feature matrix.
        w: Weight matrix.
        b: Bias vector.

    Returns:
        Dictionary containing the pre-activation matrix.
    """
    return {
        "Z": x @ w + b,
    }
