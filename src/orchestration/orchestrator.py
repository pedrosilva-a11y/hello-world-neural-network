"""Pipeline orchestrator for the Digit Recognizer project."""

import numpy as np

from data_loading.data_loading import load_digit_recognizer_data
from preprocessing.preprocessing import preprocess_digit_recognizer_data

DigitRecognizerPreprocessedData = tuple[np.ndarray, np.ndarray, np.ndarray]


def run_digit_recognizer_preprocessing_pipeline() -> DigitRecognizerPreprocessedData:
    """Run the data-loading and preprocessing steps in sequence.

    Returns:
        Tuple containing the training feature matrix, training label array,
        and testing feature matrix.
    """
    loaded_data = load_digit_recognizer_data()

    return preprocess_digit_recognizer_data(loaded_data=loaded_data)


def main() -> None:
    """Run the current Digit Recognizer pipeline and print the output shapes."""
    print("Running Digit Recognizer pipeline...")

    x_train_matrix, y_train_array, x_test_matrix = (
        run_digit_recognizer_preprocessing_pipeline()
    )

    print("Pipeline completed.")
    print("Train feature matrix shape:", x_train_matrix.shape)
    print("Train label array shape:", y_train_array.shape)
    print("Test feature matrix shape:", x_test_matrix.shape)


if __name__ == "__main__":
    main()
