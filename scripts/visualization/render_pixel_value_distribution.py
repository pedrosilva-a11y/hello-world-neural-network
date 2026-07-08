"""Utilities for rendering pixel intensity distributions."""

import matplotlib.pyplot as plt


def render_pixel_value_distribution(
    distribution: dict[int, int],
    title: str,
    use_log_scale: bool = False,
) -> None:
    """Render a pixel intensity distribution as a bar chart.

    Args:
        distribution: Dictionary mapping pixel intensity values to counts.
        title: Plot title.
        use_log_scale: Whether to use logarithmic scale on the y-axis.
    """
    pixel_values = list(distribution.keys())
    counts = list(distribution.values())

    plt.figure(figsize=(12, 5))
    plt.bar(pixel_values, counts)
    plt.title(title)
    plt.xlabel("Pixel intensity value")
    plt.ylabel("Count")

    if use_log_scale:
        plt.yscale("log")

    plt.tight_layout()
    plt.show()
