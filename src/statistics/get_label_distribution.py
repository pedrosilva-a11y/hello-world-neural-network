"""Label distribution statistics."""

from typing import TypedDict

import numpy as np


class LabelDistribution(TypedDict):
    """Label distribution summary."""

    total_examples: int
    counts: dict[str, int]
    percentages: dict[str, float]


def get_label_distribution(y: np.ndarray) -> LabelDistribution:
    """Compute label counts and percentages.

    Args:
        y: Label array.

    Returns:
        Dictionary containing total examples, label counts, and label percentages.

    Raises:
        ValueError: If the label array is empty.
    """
    if y.size == 0:
        raise ValueError("Label array must not be empty.")

    labels, counts = np.unique(y, return_counts=True)
    total_examples = int(y.shape[0])

    count_distribution = {
        str(int(label)): int(count)
        for label, count in zip(labels, counts, strict=True)
    }
    percentage_distribution = {
        str(int(label)): float((count / total_examples) * 100)
        for label, count in zip(labels, counts, strict=True)
    }

    return {
        "total_examples": total_examples,
        "counts": count_distribution,
        "percentages": percentage_distribution,
    }
