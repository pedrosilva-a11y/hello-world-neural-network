"""Script for rendering Digit Recognizer images."""

import matplotlib.pyplot as plt
import pandas as pd


def render_entry_by_index(df_pixels: pd.DataFrame, target_index: int) -> None:
    """Render one flattened 28x28 digit image by row index.

    Args:
        df_pixels: DataFrame containing flattened pixel values.
        target_index: Row index of the image to render.

    Raises:
        ValueError: If the target index is outside the DataFrame range.
    """
    if target_index < 0 or target_index >= len(df_pixels):
        raise ValueError(
            f"Invalid index {target_index}. Valid range is 0 to {len(df_pixels) - 1}."
        )

    flat_pixels = df_pixels.iloc[target_index].to_numpy()
    image_matrix = flat_pixels.reshape(28, 28)

    print(f"Rendering entry at index: {target_index}...")

    plt.imshow(image_matrix, cmap="gray", interpolation="nearest")
    plt.title(f"MNIST Entry - Global Index {target_index}")
    plt.axis("off")
    plt.show()
