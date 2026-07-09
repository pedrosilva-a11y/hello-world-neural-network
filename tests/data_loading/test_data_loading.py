"""Tests for the Digit Recognizer data-loading step."""

import unittest
from unittest.mock import call, patch

import pandas as pd

from data_loading import data_loading


class TestDataLoading(unittest.TestCase):
    """Tests for the data-loading orchestration module."""

    def test_load_digit_recognizer_data_reads_and_splits_train_and_test_sets(self) -> None:
        """Load datasets and split training features from labels."""
        data_loading_config = {
            "training_path": "data/train.csv",
            "testing_path": "data/test.csv",
            "label_column": "label",
        }

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
        x_train = pd.DataFrame(
            {
                "pixel0": [0],
                "pixel1": [255],
            }
        )
        y_train = pd.Series([7], name="label")

        with (
            patch.object(
                data_loading,
                "read_csv",
                side_effect=[df_train, df_test],
            ) as mock_read_csv,
            patch.object(
                data_loading,
                "split_features_and_labels",
                return_value=(x_train, y_train),
            ) as mock_split_features_and_labels,
        ):
            result_x_train, result_y_train, result_x_test = (
                data_loading.load_digit_recognizer_data(
                    data_loading_config=data_loading_config,
                )
            )

        self.assertIs(result_x_train, x_train)
        self.assertIs(result_y_train, y_train)
        self.assertIs(result_x_test, df_test)

        mock_read_csv.assert_has_calls(
            [
                call(file_path=data_loading.ROOT_DIR / "data/train.csv"),
                call(file_path=data_loading.ROOT_DIR / "data/test.csv"),
            ]
        )
        self.assertEqual(mock_read_csv.call_count, 2)

        mock_split_features_and_labels.assert_called_once_with(
            df=df_train,
            label_column="label",
        )

    def test_load_digit_recognizer_data_supports_custom_label_column(self) -> None:
        """Pass a custom label column from config to the splitting utility."""
        data_loading_config = {
            "training_path": "data/train.csv",
            "testing_path": "data/test.csv",
            "label_column": "target",
        }

        df_train = pd.DataFrame(
            {
                "target": [1],
                "pixel0": [0],
            }
        )
        df_test = pd.DataFrame(
            {
                "pixel0": [10],
            }
        )
        x_train = pd.DataFrame(
            {
                "pixel0": [0],
            }
        )
        y_train = pd.Series([1], name="target")

        with (
            patch.object(
                data_loading,
                "read_csv",
                side_effect=[df_train, df_test],
            ),
            patch.object(
                data_loading,
                "split_features_and_labels",
                return_value=(x_train, y_train),
            ) as mock_split_features_and_labels,
        ):
            data_loading.load_digit_recognizer_data(
                data_loading_config=data_loading_config,
            )

        mock_split_features_and_labels.assert_called_once_with(
            df=df_train,
            label_column="target",
        )

    def test_resolve_project_path_returns_absolute_path_unchanged(self) -> None:
        """Return absolute paths without resolving them against the project root."""
        absolute_path = data_loading.ROOT_DIR / "data/train.csv"

        result = data_loading.resolve_project_path(file_path=absolute_path)

        self.assertEqual(result, absolute_path)

    def test_resolve_project_path_resolves_relative_path_from_project_root(self) -> None:
        """Resolve relative paths from the project root."""
        result = data_loading.resolve_project_path(file_path="data/train.csv")

        self.assertEqual(result, data_loading.ROOT_DIR / "data/train.csv")


if __name__ == "__main__":
    unittest.main()
