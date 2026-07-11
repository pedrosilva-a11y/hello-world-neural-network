"""Tests for one-hot label representation utilities."""

import unittest

import numpy as np

from training.encoding.label_one_hot_representation import label_one_hot_representation


class TestLabelOneHotRepresentation(unittest.TestCase):
    """Tests for label_one_hot_representation."""

    def test_label_one_hot_representation_returns_expected_shape(self) -> None:
        """Return one row per label and one column per category."""
        labels = np.array([1, 0, 4])

        result = label_one_hot_representation(labels=labels)

        self.assertEqual(result.shape, (3, 10))

    def test_label_one_hot_representation_encodes_labels_correctly(self) -> None:
        """Encode each label as a one-hot row."""
        labels = np.array([1, 0, 4])

        result = label_one_hot_representation(labels=labels)

        expected = np.array(
            [
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            ],
        )

        np.testing.assert_array_equal(result, expected)

    def test_label_one_hot_representation_sets_one_active_class_per_row(self) -> None:
        """Return rows where exactly one class is active."""
        labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        result = label_one_hot_representation(labels=labels)

        row_sums = np.sum(result, axis=1)

        np.testing.assert_array_equal(row_sums, np.ones(10))

    def test_label_one_hot_representation_supports_first_and_last_class(self) -> None:
        """Encode labels at the first and last class indexes."""
        labels = np.array([0, 9])

        result = label_one_hot_representation(labels=labels)

        expected = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ],
        )

        np.testing.assert_array_equal(result, expected)

    def test_label_one_hot_representation_does_not_mutate_labels(self) -> None:
        """Keep the original labels array unchanged."""
        labels = np.array([3, 5, 7])
        original_labels = labels.copy()

        label_one_hot_representation(labels=labels)

        np.testing.assert_array_equal(labels, original_labels)


if __name__ == "__main__":
    unittest.main()
