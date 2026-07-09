"""Preprocessing step for the Kaggle Digit Recognizer datasets."""

from typing import cast

import numpy as np
import pandas as pd

from preprocessing.convert_dataframe_to_matrix import convert_dataframe_to_matrix
from preprocessing.normalize_pixels import normalize_pixels

DigitRecognizerLoadedData = tuple[pd.DataFrame, pd.Series, pd.DataFrame]
DigitRecognizerPreprocessedData = tuple[np.ndarray, np.ndarray, np.ndarray]


def preprocess_digit_recognizer_data(
    loaded_data: DigitRecognizerLoadedData,
    normalize: bool = False,
) -> DigitRecognizerPreprocessedData:
    """Preprocess the output of the Digit Recognizer data-loading step.

    This step converts loaded pandas objects into NumPy arrays and optionally
    normalizes pixel feature values from the 0-255 range to the 0-1 range.

    Args:
        loaded_data: Tuple containing training features, training labels, and
            testing features.
        normalize: Whether to normalize pixel feature values.

    Returns:
        Tuple containing the training feature matrix, training label array,
        and testing feature matrix.
    """
    x_train, y_train, x_test = loaded_data

    x_train_matrix = convert_dataframe_to_matrix(df=x_train)
    y_train_array = cast(np.ndarray, y_train.to_numpy())
    x_test_matrix = convert_dataframe_to_matrix(df=x_test)

    if normalize:
        x_train_matrix = normalize_pixels(x=x_train_matrix)
        x_test_matrix = normalize_pixels(x=x_test_matrix)

    return x_train_matrix, y_train_array, x_test_matrix
