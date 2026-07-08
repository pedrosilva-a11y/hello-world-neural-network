"""Tests for neural network prediction utilities."""

import unittest

import numpy as np

from training.prediction.get_predictions import get_predictions


class TestGetPredictions(unittest.TestCase):
    """Tests for get_predictions."""

    def test_get_predictions_returns_expected_key(self) -> None:
        """Return a dictionary containing the predictions array."""
        activation = np.array([[0.1, 0.8, 0.1]])

        result = get_predictions(activation=activation)

        self.assertEqual(set(result.keys()), {"predictions"})

    def test_get_predictions_returns_max_index_per_row(self) -> None:
        """Return the class index with the highest probability for each row."""
        activation = np.array(
            [
                [0.1, 0.8, 0.1],
                [0.7, 0.2, 0.1],
                [0.2, 0.3, 0.5],
            ],
        )

        result = get_predictions(activation=activation)

        expected_predictions = np.array([1, 0, 2])

        np.testing.assert_array_equal(result["predictions"], expected_predictions)

    def test_get_predictions_returns_one_prediction_per_example(self) -> None:
        """Return one prediction for each input example."""
        activation = np.array(
            [
                [0.1, 0.9],
                [0.6, 0.4],
                [0.3, 0.7],
                [0.8, 0.2],
            ],
        )

        result = get_predictions(activation=activation)

        self.assertEqual(result["predictions"].shape, (4,))

    def test_get_predictions_selects_first_index_when_probabilities_tie(self) -> None:
        """Return the first maximum index when two classes have equal probability."""
        activation = np.array(
            [
                [0.5, 0.5, 0.1],
                [0.2, 0.8, 0.8],
            ],
        )

        result = get_predictions(activation=activation)

        expected_predictions = np.array([0, 1])

        np.testing.assert_array_equal(result["predictions"], expected_predictions)


if __name__ == "__main__":
    unittest.main()
