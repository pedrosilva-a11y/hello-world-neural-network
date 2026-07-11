"""Gradient computations for ReLU and softmax with categorical cross-entropy."""

from typing import cast

import numpy as np


def gradient_computations_softmax(
    x: np.ndarray,
    yhot: np.ndarray,
    forward_pass_results: dict[str, np.ndarray],
    layer: int,
) -> dict[str, np.ndarray]:
    """Compute gradients for a softmax output layer.

    This function assumes the softmax activation is paired with categorical
    cross-entropy loss, which simplifies the output-layer gradient to
    A_current - yhot.

    Args:
        x: Input feature matrix. Used as the previous activation when the
            softmax layer is the first layer.
        yhot: One-hot representation of the true labels.
        forward_pass_results: Dictionary containing Z and A values for all layers.
        layer: Layer number for which gradients are being computed.

    Returns:
        Dictionary containing the layer-specific dZ, dW, and db gradients.
    """
    num_examples = x.shape[0]

    A_current = forward_pass_results[f"A{layer}"]

    A_prev = x if layer == 1 else forward_pass_results[f"A{layer - 1}"]

    dZ_current = A_current - yhot
    dW_current = A_prev.T @ dZ_current / num_examples
    db_current = np.sum(dZ_current, axis=0, keepdims=True) / num_examples

    return {
        f"dZ{layer}": dZ_current,
        f"dW{layer}": dW_current,
        f"db{layer}": db_current,
    }


def _relu_derivative(pre_activation: np.ndarray) -> np.ndarray:
    """Compute the ReLU derivative element-wise.

    Args:
        pre_activation: Pre-activation values before applying ReLU.

    Returns:
        Array containing 1 where pre_activation is positive and 0 otherwise.
    """
    return cast(np.ndarray, np.where(pre_activation > 0, 1, 0))


def gradient_computations_relu(
    x: np.ndarray,
    gradients: dict[str, np.ndarray],
    parameters: dict[str, np.ndarray],
    forward_pass_results: dict[str, np.ndarray],
    layer: int,
) -> dict[str, np.ndarray]:
    """Compute gradients for a ReLU hidden layer.

    This function propagates gradients from the next layer backward through the
    current ReLU layer and computes dA, dZ, dW, and db for the current layer.

    Args:
        x: Input feature matrix. Used as the previous activation when the
            current ReLU layer is the first layer.
        gradients: Dictionary containing gradients already computed for later layers.
        parameters: Dictionary containing weights and biases for each layer.
        forward_pass_results: Dictionary containing Z and A values for all layers.
        layer: Layer number for which gradients are being computed.

    Returns:
        Dictionary updated with the gradients of the current layer.
    """
    number_of_examples = x.shape[0]

    dZ_next = gradients[f"dZ{layer + 1}"]
    W_next = parameters[f"W{layer + 1}"]
    Z_current = forward_pass_results[f"Z{layer}"]

    A_prev = x if layer == 1 else forward_pass_results[f"A{layer - 1}"]

    dA_current = dZ_next @ W_next.T
    gradients[f"dA{layer}"] = dA_current

    dZ_current = dA_current * _relu_derivative(pre_activation=Z_current)
    gradients[f"dZ{layer}"] = dZ_current

    dW_current = A_prev.T @ dZ_current / number_of_examples
    gradients[f"dW{layer}"] = dW_current

    db_current = np.sum(dZ_current, axis=0, keepdims=True) / number_of_examples
    gradients[f"db{layer}"] = db_current

    return gradients
