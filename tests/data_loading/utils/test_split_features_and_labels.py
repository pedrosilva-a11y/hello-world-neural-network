"""Tests for the feature and label splitting utility."""

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from data_loading.utils.split_features_and_labels import split_features_and_labels


class TestSplitFeaturesAndLabels(unittest.TestCase):
    """Tests for split_features_and_labels."""

    def setUp(self) -> None:
        """Create reusable labeled DataFrame test data."""
        self.df: pd.DataFrame = pd.DataFrame(
            {
                "label": [7, 2, 9],
                "pixel0": [0, 10, 255],
                "pixel1": [100, 120, 140],
                "pixel2": [255, 0, 50],
            }
        )

    def test_split_features_and_labels_returns_features_without_label_column(self) -> None:
        """Return feature columns after removing the default label column."""
        features, _ = split_features_and_labels(df=self.df)

        expected_features = pd.DataFrame(
            {
                "pixel0": [0, 10, 255],
                "pixel1": [100, 120, 140],
                "pixel2": [255, 0, 50],
            }
        )

        assert_frame_equal(features, expected_features)

    def test_split_features_and_labels_returns_labels_from_default_label_column(self) -> None:
        """Return labels from the default label column."""
        _, labels = split_features_and_labels(df=self.df)

        expected_labels = pd.Series([7, 2, 9], name="label")

        assert_series_equal(labels, expected_labels)

    def test_split_features_and_labels_preserves_feature_column_order(self) -> None:
        """Preserve the original feature column order after removing labels."""
        features, _ = split_features_and_labels(df=self.df)

        self.assertEqual(
            list(features.columns),
            ["pixel0", "pixel1", "pixel2"],
        )

    def test_split_features_and_labels_supports_custom_label_column(self) -> None:
        """Split features and labels using a custom label column name."""
        df = pd.DataFrame(
            {
                "target": [1, 0],
                "pixel0": [50, 60],
                "pixel1": [70, 80],
            }
        )

        features, labels = split_features_and_labels(
            df=df,
            label_column="target",
        )

        expected_features = pd.DataFrame(
            {
                "pixel0": [50, 60],
                "pixel1": [70, 80],
            }
        )
        expected_labels = pd.Series([1, 0], name="target")

        assert_frame_equal(features, expected_features)
        assert_series_equal(labels, expected_labels)

    def test_split_features_and_labels_preserves_original_index(self) -> None:
        """Preserve the original DataFrame index in both outputs."""
        df = self.df.copy()
        df.index = [10, 20, 30]

        features, labels = split_features_and_labels(df=df)

        self.assertEqual(list(features.index), [10, 20, 30])
        self.assertEqual(list(labels.index), [10, 20, 30])

    def test_split_features_and_labels_does_not_mutate_original_dataframe(self) -> None:
        """Keep the original DataFrame unchanged after splitting."""
        original_df = self.df.copy()

        split_features_and_labels(df=self.df)

        assert_frame_equal(self.df, original_df)

    def test_split_features_and_labels_raises_error_when_label_column_is_missing(self) -> None:
        """Raise ValueError when the expected label column is missing."""
        df = pd.DataFrame(
            {
                "pixel0": [0, 10],
                "pixel1": [20, 30],
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "expected label column 'label' was not found",
        ):
            split_features_and_labels(df=df)


if __name__ == "__main__":
    unittest.main()
