"""Categorical cross-entropy loss function."""

import numpy as np

EPSILON = 1e-15

def categorical_cross_entropy(
    y_one_hot: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute categorical cross-entropy loss.

    Args:
        y_one_hot: One-hot representation of the true categorical labels.
        y_pred: Predicted probability distribution per input example.

    Returns:
        Average categorical cross-entropy loss.
    """
    num_examples = y_one_hot.shape[0]

    clipped_y_pred = np.clip(y_pred, EPSILON, 1.0)
    example_losses = np.sum(y_one_hot * np.log(clipped_y_pred), axis=1)
    loss = -float(np.sum(example_losses) / num_examples)

    return loss
