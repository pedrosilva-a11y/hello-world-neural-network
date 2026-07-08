"""Pixel intensity statistics for Digit Recognizer data."""

import numpy as np
import pandas as pd

MIN_PIXEL_VALUE = 0
MAX_PIXEL_VALUE = 255


def get_pixel_value_distribution(
    df_pixels: pd.DataFrame,
    target_index: int | None = None,
    include_missing_values: bool = True,
) -> dict[int, int]:
    """Count pixel intensity values across a dataset or one selected image.

    Args:
        df_pixels: DataFrame containing only pixel columns.
        target_index: Optional image row index. If omitted, all images are counted.
        include_missing_values: Whether to include all values from 0 to 255, even if count is zero.

    Returns:
        Dictionary mapping pixel intensity values to counts, ordered by pixel value.

    Raises:
        ValueError: If the DataFrame is empty, the index is invalid, or values are outside 0 to 255.
    """
    if df_pixels.empty:
        raise ValueError("Pixel DataFrame cannot be empty.")

    if target_index is None:
        pixel_values = df_pixels.to_numpy().ravel()
    else:
        if target_index < 0 or target_index >= len(df_pixels):
            raise ValueError(
                f"Invalid index {target_index}. Valid range is 0 to {len(df_pixels) - 1}."
            )

        pixel_values = df_pixels.iloc[target_index].to_numpy()

    min_value = int(np.min(pixel_values))
    max_value = int(np.max(pixel_values))

    if min_value < MIN_PIXEL_VALUE or max_value > MAX_PIXEL_VALUE:
        raise ValueError(
            f"Pixel values must be between {MIN_PIXEL_VALUE} and {MAX_PIXEL_VALUE}."
        )

    unique_values, counts = np.unique(pixel_values, return_counts=True)

    observed_distribution = {
        int(pixel_value): int(count)
        for pixel_value, count in zip(unique_values, counts, strict=True)
    }

    if include_missing_values:
        return {
            pixel_value: observed_distribution.get(pixel_value, 0)
            for pixel_value in range(MIN_PIXEL_VALUE, MAX_PIXEL_VALUE + 1)
        }

    return dict(sorted(observed_distribution.items()))
