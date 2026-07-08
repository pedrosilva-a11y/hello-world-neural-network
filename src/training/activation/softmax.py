"""Softmax activation function."""

import numpy as np


def softmax(logits: np.ndarray) -> dict[str, np.ndarray]:
    """Compute softmax probabilities from raw logits.

    Args:
        logits: Raw score matrix with shape examples by classes.

    Returns:
        Dictionary containing the softmax probability matrix.
    """
    stable_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(stable_logits)
    row_sums = np.sum(exp_logits, axis=1, keepdims=True)

    return {
        "A": exp_logits / row_sums,
    }
