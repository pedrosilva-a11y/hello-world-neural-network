"""Prediction utilities for neural network output activations."""

import numpy as np


def get_predictions(activation: np.ndarray) -> dict[str, np.ndarray]:
    """Get predicted classes from output activation probabilities.

    Args:
        activation: Output activation matrix with one probability row per example.

    Returns:
        Dictionary containing predicted class indexes for all examples.
    """
    return {
        "predictions": np.argmax(activation, axis=1),
    }
