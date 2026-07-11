"""Defines the linear pre-activation computation for a neural network layer."""

import numpy as np


def compute_pre_activation(
    a: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    layer_number: int,
) -> dict[str, np.ndarray]:
    """Compute the linear pre-activation values for a neural network layer.

    Args:
        a: Activation of the previous layer. For the first layer,
            the input features represent the previous activation values.
        w: Weight matrix of the current layer.
        b: Bias vector of the current layer.
        layer_number: Layer that pre-activation is being computed this iteration.

    Returns:
        Dictionary containing the pre-activation matrix.
    """
    return {
        f"Z{layer_number}": a @ w + b,
    }
