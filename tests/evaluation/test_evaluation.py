"""Tests for evaluation orchestration utilities."""

import unittest
from unittest.mock import patch

import numpy as np

from evaluation import evaluation


class TestRunEvaluation(unittest.TestCase):
    """Tests for run_evaluation."""

    def test_run_evaluation_coordinates_accuracy_computation(self) -> None:
        """Run accuracy computation using true labels and predicted labels."""
        y = np.array([1, 0, 4, 7])
        ypred = np.array([1, 2, 4, 7])

        with patch.object(
            evaluation,
            "compute_accuracy",
            return_value={"accuracy": 0.75},
        ) as mock_compute_accuracy:
            result = evaluation.run_evaluation(
                y=y,
                ypred=ypred,
            )

        self.assertEqual(result["accuracy"], 0.75)

        mock_compute_accuracy.assert_called_once_with(
            y=y,
            ypred=ypred,
        )

    def test_run_evaluation_returns_expected_key(self) -> None:
        """Return a dictionary containing the accuracy value."""
        y = np.array([1, 0, 4, 7])
        ypred = np.array([1, 2, 4, 7])

        result = evaluation.run_evaluation(
            y=y,
            ypred=ypred,
        )

        self.assertEqual(set(result.keys()), {"accuracy"})

    def test_run_evaluation_returns_accuracy_float(self) -> None:
        """Return accuracy as a float value."""
        y = np.array([1, 0, 4, 7])
        ypred = np.array([1, 2, 4, 7])

        result = evaluation.run_evaluation(
            y=y,
            ypred=ypred,
        )

        self.assertIsInstance(result["accuracy"], float)

    def test_run_evaluation_returns_expected_accuracy(self) -> None:
        """Return the expected accuracy value."""
        y = np.array([1, 0, 4, 7])
        ypred = np.array([1, 2, 4, 7])

        result = evaluation.run_evaluation(
            y=y,
            ypred=ypred,
        )

        self.assertEqual(result["accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()
