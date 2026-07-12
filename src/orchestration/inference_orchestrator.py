"""Inference orchestrator for the Digit Recognizer project."""

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from data_loading.data_loading import load_digit_recognizer_data
from inference.inference import run_inference
from inference.submission import save_kaggle_submission
from inference.utils.parameters_io import load_parameters
from preprocessing.preprocessing import preprocess_digit_recognizer_data
from utils.json_io import save_json

ROOT_DIR = Path(__file__).resolve().parents[2]

DigitRecognizerPreprocessedData = tuple[np.ndarray, np.ndarray, np.ndarray]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run Digit Recognizer inference from a saved experiment.",
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="Name of the saved experiment under the results directory.",
    )
    parser.add_argument(
        "--submission-file-name",
        default="submission.csv",
        help="Name of the Kaggle submission CSV file to create.",
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


def load_experiment_config(
    summary_path: Path,
) -> dict[str, Any]:
    """Load the original experiment config from a saved summary file.

    Args:
        summary_path: Path to the saved experiment summary.

    Returns:
        Original experiment configuration.

    Raises:
        FileNotFoundError: If the summary file does not exist.
        ValueError: If the summary file does not contain a valid config.
    """
    if not summary_path.exists():
        raise FileNotFoundError(f"Experiment summary not found: {summary_path}")

    with summary_path.open(mode="r", encoding="utf-8") as summary_file:
        summary = json.load(summary_file)

    if not isinstance(summary, dict):
        raise ValueError("Experiment summary must be a dictionary.")

    config = summary.get("config")

    if not isinstance(config, dict):
        raise ValueError("Experiment summary does not contain a valid config.")

    return cast(dict[str, Any], config)


def run_digit_recognizer_inference_preprocessing_pipeline(
    data_loading_config: dict[str, Any],
    preprocessing_config: dict[str, Any],
) -> DigitRecognizerPreprocessedData:
    """Run data loading and preprocessing for inference.

    Args:
        data_loading_config: Data-loading configuration section.
        preprocessing_config: Preprocessing configuration section.

    Returns:
        Tuple containing the full training feature matrix, full training label
        array, and testing feature matrix.
    """
    loaded_data = load_digit_recognizer_data(
        data_loading_config=data_loading_config,
    )

    return preprocess_digit_recognizer_data(
        loaded_data=loaded_data,
        preprocessing_config=preprocessing_config,
    )


def main() -> None:
    """Run inference and save a Kaggle submission file."""
    args = parse_args()

    experiment_name = str(args.experiment_name)
    experiment_results_dir = ROOT_DIR / "results" / experiment_name

    summary_path = experiment_results_dir / "summary.json"
    parameters_path = experiment_results_dir / "parameters.npz"
    submission_path = experiment_results_dir / str(args.submission_file_name)
    inference_summary_path = experiment_results_dir / "inference_summary.json"

    config = load_experiment_config(summary_path=summary_path)

    data_loading_config = get_config_section(
        config=config,
        section_name="data_loading",
    )
    preprocessing_config = get_config_section(
        config=config,
        section_name="preprocessing",
    )
    model_config = get_config_section(
        config=config,
        section_name="model",
    )
    outputs_config = get_config_section(
        config=config,
        section_name="outputs",
    )

    _x_full_train_matrix, _y_full_train_array, x_test_matrix = (
        run_digit_recognizer_inference_preprocessing_pipeline(
            data_loading_config=data_loading_config,
            preprocessing_config=preprocessing_config,
        )
    )

    parameters = load_parameters(file_path=parameters_path)
    neurons_profile = list(model_config["neurons_profile"])

    inference_output = run_inference(
        x_test_matrix=x_test_matrix,
        parameters=parameters,
        neurons_profile=neurons_profile,
    )

    predictions = inference_output["predictions"]

    save_kaggle_submission(
        predictions=predictions,
        file_path=submission_path,
    )

    prediction_preview_size = int(outputs_config["prediction_preview_size"])

    inference_summary: dict[str, object] = {
        "experiment_name": experiment_name,
        "metadata": {
            "summary_path": str(summary_path),
            "parameters_path": str(parameters_path),
            "submission_path": str(submission_path),
            "model": {
                "name": model_config["name"],
                "neurons_profile": model_config["neurons_profile"],
            },
            "data_shapes": {
                "test": list(x_test_matrix.shape),
            },
        },
        "outputs": {
            "test_predictions_preview": predictions[
                :prediction_preview_size
            ].tolist(),
            "num_predictions": int(predictions.shape[0]),
        },
    }

    save_json(
        data=inference_summary,
        file_path=inference_summary_path,
    )

    print("Experiment name:")
    print(experiment_name)

    print("Number of test predictions:")
    print(predictions.shape[0])

    print("Submission saved to:")
    print(submission_path)

    print("Inference summary saved to:")
    print(inference_summary_path)


if __name__ == "__main__":
    main()
