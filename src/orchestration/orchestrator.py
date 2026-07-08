"""Pipeline orchestrator for the Digit Recognizer project."""

import pandas as pd

from data_loading.data_loading import load_digit_recognizer_data
from preprocessing.preprocessing import preprocess_digit_recognizer_data


def run_digit_recognizer_preprocessing_pipeline() -> pd.DataFrame:
    """Run the data-loading and preprocessing steps in sequence.

    Returns:
        Combined pixel DataFrame produced by the preprocessing step.
    """
    loaded_data = load_digit_recognizer_data()

    return preprocess_digit_recognizer_data(loaded_data=loaded_data)


def main() -> None:
    """Run the current Digit Recognizer pipeline and print the output shape."""
    print("Running Digit Recognizer pipeline...")

    df_combined = run_digit_recognizer_preprocessing_pipeline()

    print("Pipeline completed.")
    print("Combined pixel data shape:", df_combined.shape)


if __name__ == "__main__":
    main()
