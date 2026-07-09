"""Pipeline orchestrator for the Digit Recognizer project."""

import argparse
from pathlib import Path
from statistics.get_label_distribution import get_label_distribution
from typing import Any, cast

import numpy as np

from data_loading.data_loading import load_digit_recognizer_data
from data_splitting.data_splitting import split_digit_recognizer_training_data
from preprocessing.preprocessing import preprocess_digit_recognizer_data
from training.training import run_training_iterations
from utils.json_io import save_json
from utils.yaml_io import read_yaml

ROOT_DIR = Path(__file__).resolve().parents[2]

DigitRecognizerPreprocessedData = tuple[np.ndarray, np.ndarray, np.ndarray]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the Digit Recognizer pipeline from a YAML config.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the experiment YAML config file.",
    )

    return parser.parse_args()


def get_config_section(
    config: dict[str, Any],
    section_name: str,
) -> dict[str, Any]:
    """Get a required top-level config section.

    Args:
        config: Full experiment configuration.
        section_name: Name of the required section.

    Returns:
        Requested config section.

    Raises:
        ValueError: If the section is missing or is not a dictionary.
    """
    section = config.get(section_name)

    if not isinstance(section, dict):
        msg = f"Missing or invalid config section: {section_name}"
        raise ValueError(msg)

    return cast(dict[str, Any], section)


def run_digit_recognizer_preprocessing_pipeline(
    data_loading_config: dict[str, Any],
    preprocessing_config: dict[str, Any],
) -> DigitRecognizerPreprocessedData:
    """Run the data-loading and preprocessing steps in sequence.

    Args:
        data_loading_config: Data-loading configuration section.
        preprocessing_config: Preprocessing configuration section.

    Returns:
        Tuple containing the training feature matrix, training label array,
        and testing feature matrix.
    """
    loaded_data = load_digit_recognizer_data(
        data_loading_config=data_loading_config,
    )

    return preprocess_digit_recognizer_data(
        loaded_data=loaded_data,
        preprocessing_config=preprocessing_config,
    )


def main() -> None:
    """Run the Digit Recognizer pipeline and save experiment summary."""
    args = parse_args()
    config = read_yaml(file_path=args.config)

    experiment_name = str(config["experiment_name"])

    data_loading_config = get_config_section(
        config=config,
        section_name="data_loading",
    )

    preprocessing_config = get_config_section(
        config=config,
        section_name="preprocessing",
    )
    data_splitting_config = get_config_section(
        config=config,
        section_name="data_splitting",
    )
    model_config = get_config_section(
        config=config,
        section_name="model",
    )
    training_config = get_config_section(
        config=config,
        section_name="training",
    )
    outputs_config = get_config_section(
        config=config,
        section_name="outputs",
    )

    x_full_train_matrix, y_full_train_array, x_test_matrix = (
        run_digit_recognizer_preprocessing_pipeline(
            data_loading_config=data_loading_config,
            preprocessing_config=preprocessing_config,
        )
    )

    split_output = split_digit_recognizer_training_data(
        x=x_full_train_matrix,
        y=y_full_train_array,
        data_splitting_config=data_splitting_config,
    )

    x_train_matrix = split_output["x_train"]
    y_train_array = split_output["y_train"]
    x_validation_matrix = split_output["x_validation"]
    y_validation_array = split_output["y_validation"]

    training_output = run_training_iterations(
        x_train=x_train_matrix,
        y_train=y_train_array,
        x_validation=x_validation_matrix,
        y_validation=y_validation_array,
        model_config=model_config,
        training_config=training_config,
    )

    prediction_preview_size = int(outputs_config["prediction_preview_size"])

    summary: dict[str, object] = {
        "experiment_name": experiment_name,
        "config": config,
        "metadata": {
            "config_path": str(args.config),
            "data_shapes": {
                "full_train": list(x_full_train_matrix.shape),
                "train": list(x_train_matrix.shape),
                "validation": list(x_validation_matrix.shape),
                "test": list(x_test_matrix.shape),
            },
        },
        "metrics": {
            "train_loss": training_output["train_loss"],
            "train_accuracy": training_output["train_accuracy"],
            "validation_loss": training_output["validation_loss"],
            "validation_accuracy": training_output["validation_accuracy"],
        },
        "data_distribution": {
            "full_train": get_label_distribution(y=y_full_train_array),
            "train": get_label_distribution(y=y_train_array),
            "validation": get_label_distribution(y=y_validation_array),
        },
        "outputs": {
            "train_predictions_preview": training_output["train_predictions"][
                :prediction_preview_size
            ].tolist(),
            "validation_predictions_preview": training_output["validation_predictions"][
                :prediction_preview_size
            ].tolist(),
            "validation_labels_preview": y_validation_array[
                :prediction_preview_size
            ].tolist(),
        },
    }

    summary_path = ROOT_DIR / "results" / experiment_name / "summary.json"

    save_json(
        data=summary,
        file_path=summary_path,
    )

    print("Experiment name:")
    print(experiment_name)

    print("Final training categorical cross-entropy loss:")
    print(training_output["train_loss"][-1])

    print("Final training accuracy:")
    print(training_output["train_accuracy"][-1])

    print("Final validation categorical cross-entropy loss:")
    print(training_output["validation_loss"][-1])

    print("Final validation accuracy:")
    print(training_output["validation_accuracy"][-1])

    print("Summary saved to:")
    print(summary_path)


if __name__ == "__main__":
    main()
