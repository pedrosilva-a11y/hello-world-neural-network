"""Visualize train and validation digit label distributions."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from utils.json_io import read_json

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_NAME = "experiment_1"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Plot train and validation digit label distributions.",
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help="Name of the experiment folder under results/.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window after saving the image.",
    )

    return parser.parse_args()


def get_distribution_percentages(
    summary: dict[str, object],
    split_name: str,
) -> dict[str, float]:
    """Extract label percentage distribution for a split.

    Args:
        summary: Experiment summary loaded from JSON.
        split_name: Data split name, such as train or validation.

    Returns:
        Dictionary mapping label strings to percentage values.

    Raises:
        ValueError: If the distribution section is missing or invalid.
    """
    data_distribution = summary.get("data_distribution")

    if not isinstance(data_distribution, dict):
        raise ValueError("Expected summary.json to contain a 'data_distribution' object.")

    split_distribution = data_distribution.get(split_name)

    if not isinstance(split_distribution, dict):
        raise ValueError(f"Expected data_distribution to contain '{split_name}'.")

    percentages = split_distribution.get("percentages")

    if not isinstance(percentages, dict):
        raise ValueError(
            f"Expected data_distribution.{split_name} to contain 'percentages'."
        )

    distribution: dict[str, float] = {}

    for label, percentage in percentages.items():
        if not isinstance(label, str):
            raise ValueError("Expected distribution labels to be strings.")

        if not isinstance(percentage, int | float):
            raise ValueError("Expected distribution percentages to be numeric.")

        distribution[label] = float(percentage)

    return distribution


def plot_label_distribution(
    experiment_name: str,
    show: bool = False,
) -> Path:
    """Plot train and validation digit label distributions.

    Args:
        experiment_name: Name of the experiment folder under results/.
        show: Whether to display the plot window after saving.

    Returns:
        Path where the plot image was saved.
    """
    summary_path = ROOT_DIR / "results" / experiment_name / "summary.json"
    output_path = ROOT_DIR / "results" / experiment_name / "label_distribution.png"

    summary = read_json(file_path=summary_path)

    train_percentages = get_distribution_percentages(
        summary=summary,
        split_name="train",
    )
    validation_percentages = get_distribution_percentages(
        summary=summary,
        split_name="validation",
    )

    labels = [str(label) for label in range(10)]
    train_values = [train_percentages.get(label, 0.0) for label in labels]
    validation_values = [validation_percentages.get(label, 0.0) for label in labels]

    x_positions = list(range(len(labels)))
    bar_width = 0.4

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.bar(
        [position - bar_width / 2 for position in x_positions],
        train_values,
        width=bar_width,
        label="Train",
    )
    axis.bar(
        [position + bar_width / 2 for position in x_positions],
        validation_values,
        width=bar_width,
        label="Validation",
    )

    axis.set_title(f"Digit label distribution: {experiment_name}")
    axis.set_xlabel("Digit")
    axis.set_ylabel("Percentage")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels)
    axis.legend()

    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)

    if show:
        plt.show()

    plt.close(figure)

    return output_path


def main() -> None:
    """Plot train and validation label distributions from summary JSON."""
    args = parse_args()

    output_path = plot_label_distribution(
        experiment_name=args.experiment_name,
        show=args.show,
    )

    print("Label distribution plot saved to:")
    print(output_path)


if __name__ == "__main__":
    main()
