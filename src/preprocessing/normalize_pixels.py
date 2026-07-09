"""Normalize pixel values for image feature matrices."""

import numpy as np

MAX_PIXEL_VALUE = 255.0


def normalize_pixels(x: np.ndarray) -> np.ndarray:
    """Scale pixel values from the 0-255 range to the 0-1 range.

    Args:
        x: Input feature matrix containing raw pixel values.

    Returns:
        Normalized feature matrix.
    """
    return x.astype(float) / MAX_PIXEL_VALUE
