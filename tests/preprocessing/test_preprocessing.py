"""Tests for the Digit Recognizer preprocessing step."""

import unittest
from unittest.mock import call, patch

import numpy as np
import pandas as pd

from preprocessing import preprocessing


class TestPreprocessing(unittest.TestCase):
    """Tests for the preprocessing orchestration module."""

    def test_preprocess_digit_recognizer_data_converts_loaded_data_to_arrays(self) -> None:
        """Convert loaded feature DataFrames and label Series into NumPy arrays."""
        x_train = pd.DataFrame(
            {
                "pixel0": [0],
                "pixel1": [255],
            }
        )
        y_train = pd.Series([7], name="label")
        x_test = pd.DataFrame(
            {
                "pixel0": [10],
                "pixel1": [120],
            }
        )

        x_train_matrix = np.array([[0, 255]])
        x_test_matrix = np.array([[10, 120]])

        with patch.object(
            preprocessing,
            "convert_dataframe_to_matrix",
            side_effect=[x_train_matrix, x_test_matrix],
        ) as mock_convert_dataframe_to_matrix:
            result_x_train_matrix, result_y_train_array, result_x_test_matrix = (
                preprocessing.preprocess_digit_recognizer_data(
                    loaded_data=(x_train, y_train, x_test),
                )
            )

        self.assertIs(result_x_train_matrix, x_train_matrix)
        np.testing.assert_array_equal(result_y_train_array, np.array([7]))
        self.assertIs(result_x_test_matrix, x_test_matrix)

        mock_convert_dataframe_to_matrix.assert_has_calls(
            [
                call(df=x_train),
                call(df=x_test),
            ]
        )
        self.assertEqual(mock_convert_dataframe_to_matrix.call_count, 2)


if __name__ == "__main__":
    unittest.main()
