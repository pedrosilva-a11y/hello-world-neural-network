"""Prediction utilities for neural network output activations."""

from typing import cast

import numpy as np


def get_predictions(activation: np.ndarray) -> np.ndarray:
    """Get predicted classes from output activation probabilities.

    Args:
        activation: Output activation matrix with one probability row per example.

    Returns:
        NumPy array containing the predicted class indexes for all examples.
    """
    return cast(np.ndarray, np.argmax(activation, axis=1))
