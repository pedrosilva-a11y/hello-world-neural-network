"""Normalize pixel values for image feature matrices."""

import numpy as np


def normalize_pixels(
    x: np.ndarray,
    pixel_scale_value: float = 255.0,
) -> np.ndarray:
    """Scale pixel values from the 0-255 range to the 0-1 range.

    Args:
        x: Input feature matrix containing raw pixel values.
        pixel_scale_value: Maximum pixel value used as the normalization divisor.

    Returns:
        Normalized feature matrix.
    """
    return x.astype(float) / pixel_scale_value
