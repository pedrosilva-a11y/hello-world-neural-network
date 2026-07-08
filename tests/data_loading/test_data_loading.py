"""Tests for the Digit Recognizer data-loading step."""

import unittest
from unittest.mock import call, patch

import pandas as pd

from data_loading import data_loading


class TestDataLoading(unittest.TestCase):
    """Tests for the data-loading orchestration module."""

    def test_load_digit_recognizer_data_reads_train_and_test_sets(self) -> None:
        """Load train and test DataFrames using the configured dataset paths."""
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

        with patch.object(
            data_loading,
            "read_csv",
            side_effect=[df_train, df_test],
        ) as mock_read_csv:
            result_train, result_test = data_loading.load_digit_recognizer_data()

        self.assertIs(result_train, df_train)
        self.assertIs(result_test, df_test)

        mock_read_csv.assert_has_calls(
            [
                call(file_path=data_loading.TRAIN_SET_PATH),
                call(file_path=data_loading.TEST_SET_PATH),
            ]
        )
        self.assertEqual(mock_read_csv.call_count, 2)


if __name__ == "__main__":
    unittest.main()
