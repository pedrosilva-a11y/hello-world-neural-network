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
        description="Plot metrics for a saved experiment.",
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help="Name of the experiment folder under results/.",
    )
    parser.add_argument(
        "--split-metrics",
        action="store_true",
        help="Plot train/validation split metrics.",
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


def get_optional_metric_history(
    summary: dict[str, object],
    metric_name: str,
) -> list[float] | None:
    """Extract an optional metric history list from the experiment summary.

    Args:
        summary: Experiment summary loaded from JSON.
        metric_name: Name of the metric to extract.

    Returns:
        Metric history as a list of floats, or None when the metric is absent.
    """
    metrics = summary.get("metrics")

    if not isinstance(metrics, dict):
        raise ValueError("Expected summary.json to contain a 'metrics' object.")

    if metric_name not in metrics:
        return None

    return get_metric_history(
        summary=summary,
        metric_name=metric_name,
    )


def validate_equal_lengths(metric_histories: dict[str, list[float]]) -> None:
    """Validate that all metric histories have the same length.

    Args:
        metric_histories: Dictionary of metric names mapped to metric histories.

    Raises:
        ValueError: If metric histories have different lengths.
    """
    lengths = {metric_name: len(history) for metric_name, history in metric_histories.items()}

    if len(set(lengths.values())) != 1:
        raise ValueError(f"Metric histories must have the same length. Got: {lengths}")


def plot_standard_experiment_metrics(
    summary: dict[str, object],
    experiment_name: str,
    show: bool,
) -> Path:
    """Plot loss and accuracy history for a non-split experiment.

    Args:
        summary: Experiment summary loaded from JSON.
        experiment_name: Name of the experiment folder under results/.
        show: Whether to display the plot window after saving.

    Returns:
        Path where the plot image was saved.
    """
    output_path = ROOT_DIR / "results" / experiment_name / "metrics.png"

    loss_history = get_metric_history(
        summary=summary,
        metric_name="loss",
    )
    accuracy_history = get_metric_history(
        summary=summary,
        metric_name="accuracy",
    )

    validate_equal_lengths(
        {
            "loss": loss_history,
            "accuracy": accuracy_history,
        },
    )

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


def plot_split_experiment_metrics(
    summary: dict[str, object],
    experiment_name: str,
    show: bool,
) -> Path:
    """Plot train and validation metrics for a split experiment.

    Args:
        summary: Experiment summary loaded from JSON.
        experiment_name: Name of the experiment folder under results/.
        show: Whether to display the plot window after saving.

    Returns:
        Path where the plot image was saved.
    """
    output_path = ROOT_DIR / "results" / experiment_name / "metrics.png"

    train_loss_history = get_metric_history(
        summary=summary,
        metric_name="train_loss",
    )
    train_accuracy_history = get_metric_history(
        summary=summary,
        metric_name="train_accuracy",
    )
    validation_accuracy_history = get_metric_history(
        summary=summary,
        metric_name="validation_accuracy",
    )
    validation_loss_history = get_optional_metric_history(
        summary=summary,
        metric_name="validation_loss",
    )

    metric_histories = {
        "train_loss": train_loss_history,
        "train_accuracy": train_accuracy_history,
        "validation_accuracy": validation_accuracy_history,
    }

    if validation_loss_history is not None:
        metric_histories["validation_loss"] = validation_loss_history

    validate_equal_lengths(metric_histories=metric_histories)

    iterations = list(range(1, len(train_loss_history) + 1))

    figure, loss_axis = plt.subplots(figsize=(10, 6))

    loss_axis.plot(
        iterations,
        train_loss_history,
        marker="o",
        label="Train Loss",
    )

    if validation_loss_history is not None:
        loss_axis.plot(
            iterations,
            validation_loss_history,
            marker="o",
            linestyle=":",
            label="Validation Loss",
        )

    loss_axis.set_xlabel("Iteration")
    loss_axis.set_ylabel("Loss")

    accuracy_axis = loss_axis.twinx()
    accuracy_axis.plot(
        iterations,
        train_accuracy_history,
        marker="o",
        linestyle="--",
        label="Train Accuracy",
    )
    accuracy_axis.plot(
        iterations,
        validation_accuracy_history,
        marker="o",
        linestyle="-.",
        label="Validation Accuracy",
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


def plot_experiment_metrics(
    experiment_name: str,
    split_metrics: bool = False,
    show: bool = False,
) -> Path:
    """Plot metric history for an experiment.

    Args:
        experiment_name: Name of the experiment folder under results/.
        split_metrics: Whether to plot train/validation split metrics.
        show: Whether to display the plot window after saving.

    Returns:
        Path where the plot image was saved.
    """
    summary_path = ROOT_DIR / "results" / experiment_name / "summary.json"
    summary = read_json(file_path=summary_path)

    if split_metrics:
        return plot_split_experiment_metrics(
            summary=summary,
            experiment_name=experiment_name,
            show=show,
        )

    return plot_standard_experiment_metrics(
        summary=summary,
        experiment_name=experiment_name,
        show=show,
    )


def main() -> None:
    """Plot experiment metrics from a saved summary JSON."""
    args = parse_args()

    output_path = plot_experiment_metrics(
        experiment_name=args.experiment_name,
        split_metrics=args.split_metrics,
        show=args.show,
    )

    print("Experiment metrics plot saved to:")
    print(output_path)


if __name__ == "__main__":
    main()
