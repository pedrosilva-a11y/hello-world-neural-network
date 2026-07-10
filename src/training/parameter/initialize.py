"""Weight and bias parameter initialization utilities."""

import math

import numpy as np


def _create_weights_ndarray(
    neurons_before: int,
    neurons_current: int,
) -> np.ndarray:
    """Supporting method to compute the weights for each layer.

    Args:
        neurons_before: Quantity of neurons of the layer just before the current one.
            For the first hidden layer is the amount of features.
        neurons_current: Quantity of neurons of the current layer.

    Returns:
        The weight parameters for the pre-activation of the current layer.
    """
    return np.random.randn(neurons_before, neurons_current) * math.sqrt(2 / (neurons_before))

def _create_bias_ndarray(neurons_current: int) -> np.ndarray:
    """Supporting method to compute the bias for each layer.

    Args:
        neurons_current: Quantity of neurons of the current layer.

    Returns:
        The bias parameters for the pre-activation of the current layer.
    """
    return np.zeros((1, neurons_current))

def initialize_weights_and_bias(
    x_train: np.ndarray,
    neurons_profile: list[int],
) -> dict[str, np.ndarray]:
    """Initialize the weight matrix and bias vector for all neural network layer.

    Args:
        x_train: NumPy array containing the input features.
        neurons_profile: List containing the number of neurons per layer, in order.

    Returns:
        Dictionary containing the initialized weight matrix and bias vector.
    """
    if not neurons_profile:
        raise ValueError("neurons_profile must contain at least one layer.")

    input_size = x_train.shape[1]

    sizes = [input_size, *neurons_profile]

    parameters_mapping: dict[str, np.ndarray] = { }

    for i in range(len(neurons_profile)):
        parameters_mapping[f"W{i+1}"] = _create_weights_ndarray(sizes[i], sizes[i + 1])
        parameters_mapping[f"b{i+1}"] = _create_bias_ndarray(neurons_profile[i])

    return parameters_mapping
