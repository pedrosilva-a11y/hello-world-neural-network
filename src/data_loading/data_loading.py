"""Data-loading step for the Kaggle Digit Recognizer datasets."""

from pathlib import Path

import pandas as pd

from data_loading.utils.read_csv import read_csv
from data_loading.utils.split_features_and_labels import split_features_and_labels

ROOT_DIR = Path(__file__).resolve().parents[2]

TRAIN_SET_PATH = ROOT_DIR / "data" / "train.csv"
TEST_SET_PATH = ROOT_DIR / "data" / "test.csv"

DigitRecognizerData = tuple[pd.DataFrame, pd.Series, pd.DataFrame]


def load_digit_recognizer_data() -> DigitRecognizerData:
    """Load the Kaggle Digit Recognizer train and test datasets.

    Returns:
        A tuple containing training pixel features, training labels, and testing pixel features.
    """
    df_train = read_csv(file_path=TRAIN_SET_PATH)
    df_test = read_csv(file_path=TEST_SET_PATH)

    x_train, y_train = split_features_and_labels(df=df_train)

    return x_train, y_train, df_test


def main() -> None:
    """Load the datasets and print their shapes for manual inspection."""
    print("Loading data...")

    x_train, y_train, x_test = load_digit_recognizer_data()

    print("Train pixel shape:", x_train.shape)
    print("Train label shape:", y_train.shape)
    print("Test pixel shape:", x_test.shape)


if __name__ == "__main__":
    main()
