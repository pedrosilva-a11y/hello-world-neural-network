"""Data-loading step for the Kaggle Digit Recognizer datasets."""

from pathlib import Path

import pandas as pd

from data_loading.utils.read_csv import read_csv

ROOT_DIR = Path(__file__).resolve().parents[2]

TRAIN_SET_PATH = ROOT_DIR / "data" / "train.csv"
TEST_SET_PATH = ROOT_DIR / "data" / "test.csv"


def load_digit_recognizer_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the Kaggle Digit Recognizer train and test datasets.

    Returns:
        A tuple containing the training DataFrame and testing DataFrame.
    """
    df_train = read_csv(file_path=TRAIN_SET_PATH)
    df_test = read_csv(file_path=TEST_SET_PATH)

    return df_train, df_test


def main() -> None:
    """Load the datasets and print their shapes for manual inspection."""
    print("Loading data...")

    df_train, df_test = load_digit_recognizer_data()

    print("Train shape:", df_train.shape)
    print("Test shape:", df_test.shape)


if __name__ == "__main__":
    main()
