"""Preprocessing step for the Kaggle Digit Recognizer datasets."""

import numpy as np
import pandas as pd

from preprocessing.convert_dataframe_to_matrix import convert_dataframe_to_matrix

DigitRecognizerDataFrames = tuple[pd.DataFrame, pd.DataFrame]
DigitRecognizerMatrices = tuple[np.ndarray, np.ndarray]


def preprocess_digit_recognizer_data(
    loaded_data: DigitRecognizerDataFrames,
) -> DigitRecognizerMatrices:
    """Preprocess the output of the Digit Recognizer data-loading step.

    For now, this step converts the loaded train and test DataFrames into NumPy matrices.

    Args:
        loaded_data: Tuple containing the training DataFrame and testing DataFrame.

    Returns:
        Tuple containing the training matrix and testing matrix.
    """
    df_train, df_test = loaded_data

    training_matrix = convert_dataframe_to_matrix(df=df_train)
    testing_matrix = convert_dataframe_to_matrix(df=df_test)

    return training_matrix, testing_matrix
