"""Visualize experiment loss and accuracy history."""

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
        description="Plot loss and accuracy for a saved experiment.",
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


def get_metric_history(
    summary: dict[str, object],
    metric_name: str,
) -> list[float]:
    """Extract a metric history list from the experiment summary.

    Args:
        summary: Experiment summary loaded from JSON.
        metric_name: Name of the metric to extract.

    Returns:
        Metric history as a list of floats.

    Raises:
        ValueError: If the metric is missing or has an invalid format.
    """
    metrics = summary.get("metrics")

    if not isinstance(metrics, dict):
        raise ValueError("Expected summary.json to contain a 'metrics' object.")

    values = metrics.get(metric_name)

    if not isinstance(values, list):
        raise ValueError(f"Expected metric '{metric_name}' to be a list.")

    history: list[float] = []

    for value in values:
        if not isinstance(value, int | float):
            raise ValueError(f"Metric '{metric_name}' contains a non-numeric value.")

        history.append(float(value))

    return history


def plot_experiment_metrics(
    experiment_name: str,
    show: bool = False,
) -> Path:
    """Plot loss and accuracy history for an experiment.

    Args:
        experiment_name: Name of the experiment folder under results/.
        show: Whether to display the plot window after saving.

    Returns:
        Path where the plot image was saved.

    Raises:
        ValueError: If loss and accuracy histories have different lengths.
    """
    summary_path = ROOT_DIR / "results" / experiment_name / "summary.json"
    output_path = ROOT_DIR / "results" / experiment_name / "metrics.png"

    summary = read_json(file_path=summary_path)

    loss_history = get_metric_history(
        summary=summary,
        metric_name="loss",
    )
    accuracy_history = get_metric_history(
        summary=summary,
        metric_name="accuracy",
    )

    if len(loss_history) != len(accuracy_history):
        raise ValueError("Loss and accuracy histories must have the same length.")

    iterations = list(range(1, len(loss_history) + 1))

    figure, loss_axis = plt.subplots(figsize=(10, 6))

    loss_axis.plot(
        iterations,
        loss_history,
        marker="o",
        label="Loss",
    )
    loss_axis.set_xlabel("Iteration")
    loss_axis.set_ylabel("Loss")

    accuracy_axis = loss_axis.twinx()
    accuracy_axis.plot(
        iterations,
        accuracy_history,
        marker="o",
        linestyle="--",
        label="Accuracy",
    )
    accuracy_axis.set_ylabel("Accuracy")

    loss_lines, loss_labels = loss_axis.get_legend_handles_labels()
    accuracy_lines, accuracy_labels = accuracy_axis.get_legend_handles_labels()

    loss_axis.legend(
        loss_lines + accuracy_lines,
        loss_labels + accuracy_labels,
        loc="best",
    )

    figure.suptitle(f"Experiment metrics: {experiment_name}")
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)

    if show:
        plt.show()

    plt.close(figure)

    return output_path


def main() -> None:
    """Plot experiment metrics from a saved summary JSON."""
    args = parse_args()

    output_path = plot_experiment_metrics(
        experiment_name=args.experiment_name,
        show=args.show,
    )

    print("Experiment metrics plot saved to:")
    print(output_path)


if __name__ == "__main__":
    main()
