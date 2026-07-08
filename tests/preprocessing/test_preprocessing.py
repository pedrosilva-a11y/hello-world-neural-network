"""Tests for the Digit Recognizer preprocessing step."""

import unittest
from unittest.mock import call, patch

import numpy as np
import pandas as pd

from preprocessing import preprocessing


class TestPreprocessing(unittest.TestCase):
    """Tests for the preprocessing orchestration module."""

    def test_preprocess_digit_recognizer_data_converts_loaded_data_to_matrices(self) -> None:
        """Convert train and test DataFrames into NumPy matrices."""
        df_train = pd.DataFrame(
            {
                "label": [7],
                "pixel0": [0],
                "pixel1": [255],
            }
        )
        df_test = pd.DataFrame(
            {
                "pixel0": [10],
                "pixel1": [120],
            }
        )

        training_matrix = np.array([[7, 0, 255]])
        testing_matrix = np.array([[10, 120]])

        with patch.object(
            preprocessing,
            "convert_dataframe_to_matrix",
            side_effect=[training_matrix, testing_matrix],
        ) as mock_convert_dataframe_to_matrix:
            result_train_matrix, result_test_matrix = (
                preprocessing.preprocess_digit_recognizer_data(
                    loaded_data=(df_train, df_test),
                )
            )

        self.assertIs(result_train_matrix, training_matrix)
        self.assertIs(result_test_matrix, testing_matrix)

        mock_convert_dataframe_to_matrix.assert_has_calls(
            [
                call(df=df_train),
                call(df=df_test),
            ]
        )
        self.assertEqual(mock_convert_dataframe_to_matrix.call_count, 2)


if __name__ == "__main__":
    unittest.main()
