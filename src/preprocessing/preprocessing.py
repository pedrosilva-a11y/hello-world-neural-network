"""Preprocessing step for the Kaggle Digit Recognizer datasets."""

from typing import cast

import numpy as np
import pandas as pd

from preprocessing.convert_dataframe_to_matrix import convert_dataframe_to_matrix

DigitRecognizerLoadedData = tuple[pd.DataFrame, pd.Series, pd.DataFrame]
DigitRecognizerPreprocessedData = tuple[np.ndarray, np.ndarray, np.ndarray]


def preprocess_digit_recognizer_data(
    loaded_data: DigitRecognizerLoadedData,
) -> DigitRecognizerPreprocessedData:
    """Preprocess the output of the Digit Recognizer data-loading step.

    For now, this step converts the loaded feature DataFrames and label Series
    into NumPy arrays.

    Args:
        loaded_data: Tuple containing training features, training labels, and testing features.

    Returns:
        Tuple containing the training feature matrix, training label array,
        and testing feature matrix.
    """
    x_train, y_train, x_test = loaded_data

    x_train_matrix = convert_dataframe_to_matrix(df=x_train)
    y_train_array = cast("np.ndarray", y_train.to_numpy())
    x_test_matrix = convert_dataframe_to_matrix(df=x_test)

    return x_train_matrix, y_train_array, x_test_matrix
