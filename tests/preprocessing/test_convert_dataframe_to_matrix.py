"""Tests for the DataFrame-to-matrix preprocessing utility."""

import unittest

import numpy as np
import pandas as pd

from preprocessing.convert_dataframe_to_matrix import convert_dataframe_to_matrix


class TestConvertDataFrameToMatrix(unittest.TestCase):
    """Tests for the convert_dataframe_to_matrix preprocessing function."""

    def setUp(self) -> None:
        """Create reusable test data for DataFrame-to-matrix conversion tests."""
        self.df = pd.DataFrame(
            {
                "pixel0": [0, 50, 250],
                "pixel1": [0, 250, 0],
            },
        )
        self.expected_matrix = np.array(
            [
                [0, 0],
                [50, 250],
                [250, 0],
            ],
        )

    def test_convert_dataframe_to_matrix_returns_numpy_array(self) -> None:
        """Convert a DataFrame and return a NumPy ndarray."""
        result = convert_dataframe_to_matrix(df=self.df)

        self.assertIsInstance(result, np.ndarray)

    def test_convert_dataframe_to_matrix_preserves_shape(self) -> None:
        """Convert a DataFrame and preserve the original row and column count."""
        result = convert_dataframe_to_matrix(df=self.df)

        self.assertEqual(result.shape, (3, 2))

    def test_convert_dataframe_to_matrix_preserves_values(self) -> None:
        """Convert a DataFrame and preserve the original numeric values."""
        result = convert_dataframe_to_matrix(df=self.df)

        np.testing.assert_array_equal(result, self.expected_matrix)

    def test_convert_dataframe_to_matrix_preserves_row_as_training_example(self) -> None:
        """Keep each DataFrame row as one training example in the matrix."""
        result = convert_dataframe_to_matrix(df=self.df)

        np.testing.assert_array_equal(result[0], np.array([0, 0]))
        np.testing.assert_array_equal(result[1], np.array([50, 250]))
        np.testing.assert_array_equal(result[2], np.array([250, 0]))


if __name__ == "__main__":
    unittest.main()
