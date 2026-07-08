"""Gradient computations for softmax with categorical cross-entropy."""

import numpy as np


def gradient_computations_softmax(
    x: np.ndarray,
    yhot: np.ndarray,
    activation: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute gradients for softmax with categorical cross-entropy.

    Args:
        x: Input feature matrix.
        yhot: One-hot representation of the true labels.
        activation: Predicted probability distribution per input example.

    Returns:
        Dictionary containing dZ, dW, and db.
    """
    num_examples = x.shape[0]

    dZ = activation - yhot
    dW = x.T @ dZ / num_examples
    db = np.sum(dZ, axis=0, keepdims=True) / num_examples

    return {
        "dZ": dZ,
        "dW": dW,
        "db": db,
    }
