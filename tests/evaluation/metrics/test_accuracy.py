"""Tests for accuracy metric computation."""

import unittest

import numpy as np

from evaluation.metrics.accuracy import compute_accuracy


class TestComputeAccuracy(unittest.TestCase):
    """Tests for compute_accuracy."""

    def test_compute_accuracy_returns_expected_key(self) -> None:
        """Return a dictionary containing the accuracy value."""
        y = np.array([1, 0, 1])
        predictions = np.array([1, 1, 1])

        result = compute_accuracy(y=y, ypred=predictions)

        self.assertEqual(set(result.keys()), {"accuracy"})

    def test_compute_accuracy_returns_float(self) -> None:
        """Return accuracy as a float value."""
        y = np.array([1, 0, 1])
        predictions = np.array([1, 1, 1])

        result = compute_accuracy(y=y, ypred=predictions)

        self.assertIsInstance(result["accuracy"], float)

    def test_compute_accuracy_returns_one_for_all_correct_predictions(self) -> None:
        """Return 1.0 when all predictions match the true labels."""
        y = np.array([1, 0, 4, 7])
        predictions = np.array([1, 0, 4, 7])

        result = compute_accuracy(y=y, ypred=predictions)

        self.assertEqual(result["accuracy"], 1.0)

    def test_compute_accuracy_returns_zero_for_all_incorrect_predictions(self) -> None:
        """Return 0.0 when no predictions match the true labels."""
        y = np.array([1, 0, 4, 7])
        predictions = np.array([2, 1, 5, 8])

        result = compute_accuracy(y=y, ypred=predictions)

        self.assertEqual(result["accuracy"], 0.0)

    def test_compute_accuracy_returns_fraction_of_correct_predictions(self) -> None:
        """Return the fraction of correct predictions."""
        y = np.array([1, 0, 4, 7])
        predictions = np.array([1, 2, 4, 8])

        result = compute_accuracy(y=y, ypred=predictions)

        self.assertEqual(result["accuracy"], 0.5)

    def test_compute_accuracy_rounds_accuracy_to_two_decimal_places(self) -> None:
        """Return accuracy rounded to two decimal places."""
        y = np.array([1, 0, 4])
        predictions = np.array([1, 0, 7])

        result = compute_accuracy(y=y, ypred=predictions)

        self.assertEqual(result["accuracy"], 2/3)


if __name__ == "__main__":
    unittest.main()
