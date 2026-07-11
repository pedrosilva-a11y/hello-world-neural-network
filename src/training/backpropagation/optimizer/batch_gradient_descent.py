"""Batch gradient descent parameter update utilities."""

import numpy as np


def batch_gradient_descent(
    gradients: dict[str, np.ndarray],
    parameters: dict[str, np.ndarray],
    layer: int,
    learning_rate: float = 1e-5,
) -> dict[str, np.ndarray]:
    """Apply one batch gradient descent update to a selected layer.

    This function updates the parameter dictionary in-place for the requested
    layer. For example, when layer is 2, it updates W2 and b2 using dW2 and db2.

    Args:
        gradients: Dictionary containing layer-specific gradients.
        parameters: Dictionary containing weights and biases for each layer.
        layer: Layer number whose parameters should be updated.
        learning_rate: Step size used to update the selected layer parameters.

    Returns:
        Dictionary containing the updated parameters after applying one batch
        gradient descent step to the selected layer.
    """
    W_current = parameters[f"W{layer}"]
    dW_current = gradients[f"dW{layer}"]
    b_current = parameters[f"b{layer}"]
    db_current = gradients[f"db{layer}"]

    updated_W_current = W_current - learning_rate * dW_current
    updated_b_current = b_current - learning_rate * db_current

    parameters[f"W{layer}"] = updated_W_current
    parameters[f"b{layer}"] = updated_b_current

    return parameters
