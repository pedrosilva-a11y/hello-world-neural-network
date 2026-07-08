"""Pipeline orchestrator for the Digit Recognizer project."""

import argparse
from pathlib import Path

import numpy as np

from data_loading.data_loading import load_digit_recognizer_data
from data_splitting.data_splitting import split_digit_recognizer_training_data
from data_splitting.stratified_train_validation_split import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_VALIDATION_SIZE,
)
from preprocessing.preprocessing import preprocess_digit_recognizer_data
from training.training import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_ITERATIONS,
    run_training_iterations,
)
from utils.json_io import save_json

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_NAME = "experiment_1"
PREDICTION_PREVIEW_SIZE = 20

DigitRecognizerPreprocessedData = tuple[np.ndarray, np.ndarray, np.ndarray]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the Digit Recognizer pipeline.",
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help="Name of the experiment folder under results/.",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=DEFAULT_NUM_ITERATIONS,
        help="Number of training iterations to run.",
    )

    return parser.parse_args()


def run_digit_recognizer_preprocessing_pipeline() -> DigitRecognizerPreprocessedData:
    """Run the data-loading and preprocessing steps in sequence.

    Returns:
        Tuple containing the training feature matrix, training label array,
        and testing feature matrix.
    """
    loaded_data = load_digit_recognizer_data()

    return preprocess_digit_recognizer_data(loaded_data=loaded_data)


def main() -> None:
    """Run the Digit Recognizer pipeline and save experiment summary."""
    args = parse_args()

    x_full_train_matrix, y_full_train_array, x_test_matrix = (
        run_digit_recognizer_preprocessing_pipeline()
    )

    split_output = split_digit_recognizer_training_data(
        x=x_full_train_matrix,
        y=y_full_train_array,
        validation_size=DEFAULT_VALIDATION_SIZE,
        random_seed=DEFAULT_RANDOM_SEED,
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
        learning_rate=DEFAULT_LEARNING_RATE,
        num_iterations=args.num_iterations,
    )

    summary: dict[str, object] = {
        "experiment_name": args.experiment_name,
        "metadata": {
            "model": "single_layer_softmax_classifier",
            "num_iterations": args.num_iterations,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "validation_size": DEFAULT_VALIDATION_SIZE,
            "random_seed": DEFAULT_RANDOM_SEED,
            "full_train_shape": list(x_full_train_matrix.shape),
            "train_shape": list(x_train_matrix.shape),
            "validation_shape": list(x_validation_matrix.shape),
            "test_shape": list(x_test_matrix.shape),
        },
        "metrics": {
            "train_loss": training_output["train_loss"],
            "train_accuracy": training_output["train_accuracy"],
            "validation_accuracy": training_output["validation_accuracy"],
        },
        "outputs": {
            "train_predictions_preview": training_output["train_predictions"][
                :PREDICTION_PREVIEW_SIZE
            ].tolist(),
            "validation_predictions_preview": training_output["validation_predictions"][
                :PREDICTION_PREVIEW_SIZE
            ].tolist(),
            "validation_labels_preview": y_validation_array[
                :PREDICTION_PREVIEW_SIZE
            ].tolist(),
        },
    }

    summary_path = ROOT_DIR / "results" / args.experiment_name / "summary.json"

    save_json(
        data=summary,
        file_path=summary_path,
    )

    print("Final training categorical cross-entropy loss:")
    print(training_output["train_loss"][-1])

    print("Final training accuracy:")
    print(training_output["train_accuracy"][-1])

    print("Final validation accuracy:")
    print(training_output["validation_accuracy"][-1])

    print("Summary saved to:")
    print(summary_path)


if __name__ == "__main__":
    main()
