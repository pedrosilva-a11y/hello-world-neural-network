"""Tests for label distribution statistics."""

import unittest
from statistics.get_label_distribution import get_label_distribution

import numpy as np


class TestGetLabelDistribution(unittest.TestCase):
    """Tests for get_label_distribution."""

    def test_get_label_distribution_returns_total_examples(self) -> None:
        """Return the total number of labels."""
        y = np.array([0, 0, 1, 1, 1, 2])

        result = get_label_distribution(y=y)

        self.assertEqual(result["total_examples"], 6)

    def test_get_label_distribution_returns_label_counts(self) -> None:
        """Return counts for each label."""
        y = np.array([0, 0, 1, 1, 1, 2])

        result = get_label_distribution(y=y)

        self.assertEqual(
            result["counts"],
            {
                "0": 2,
                "1": 3,
                "2": 1,
            },
        )

    def test_get_label_distribution_returns_label_percentages(self) -> None:
        """Return percentages for each label."""
        y = np.array([0, 0, 1, 1, 1, 2])

        result = get_label_distribution(y=y)

        self.assertAlmostEqual(result["percentages"]["0"], 33.33333333333333)
        self.assertAlmostEqual(result["percentages"]["1"], 50.0)
        self.assertAlmostEqual(result["percentages"]["2"], 16.666666666666664)

    def test_get_label_distribution_supports_digit_labels(self) -> None:
        """Return string keys for digit labels."""
        y = np.array([9, 9, 8, 7])

        result = get_label_distribution(y=y)

        self.assertEqual(
            result["counts"],
            {
                "7": 1,
                "8": 1,
                "9": 2,
            },
        )

    def test_get_label_distribution_raises_error_for_empty_array(self) -> None:
        """Raise ValueError when the label array is empty."""
        y = np.array([])

        with self.assertRaisesRegex(ValueError, "Label array must not be empty."):
            get_label_distribution(y=y)


if __name__ == "__main__":
    unittest.main()
