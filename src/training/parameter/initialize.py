"""Weight and bias parameter initialization utilities."""

import numpy as np


def initialize_weights_and_bias(x_train: np.ndarray, h: int) -> dict[str, np.ndarray]:
    """Initialize the weight matrix and bias vector for one neural network layer.

    Args:
        x_train: NumPy array containing the input features.
        h: Number of neurons in the layer.

    Returns:
        Dictionary containing the initialized weight matrix and bias vector.
    """
    input_size = x_train.shape[1]

    return {
        "W1": np.random.randn(input_size, h) * (1 / np.sqrt(input_size)),
        "b1": np.zeros((1, h)),
    }
