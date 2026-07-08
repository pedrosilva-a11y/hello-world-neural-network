"""Batch gradient descent parameter update."""

import numpy as np


def batch_gradient_descent(
    W1: np.ndarray,
    b1: np.ndarray,
    dW1: np.ndarray,
    db1: np.ndarray,
    learning_rate: float = 1e-5,
) -> dict[str, np.ndarray]:
    """Update weights and bias using one step of batch gradient descent.

    Args:
        W1: Weight matrix.
        b1: Bias vector.
        dW1: Gradient of the loss with respect to W1.
        db1: Gradient of the loss with respect to b1.
        learning_rate: Step size used to update the parameters.

    Returns:
        Dictionary containing the updated W1 and b1 parameters.
    """
    updated_W1 = W1 - learning_rate * dW1
    updated_b1 = b1 - learning_rate * db1

    return {
        "W1": updated_W1,
        "b1": updated_b1,
    }
