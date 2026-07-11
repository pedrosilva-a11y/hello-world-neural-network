"""Softmax activation function."""

from typing import cast

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities from raw logits.

    Args:
        logits: Raw score matrix with shape examples by classes.

    Returns:
        Softmax probability matrix with the same shape as logits.
    """
    stable_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(stable_logits)
    row_sums = np.sum(exp_logits, axis=1, keepdims=True)

    return cast(np.ndarray, exp_logits / row_sums)
