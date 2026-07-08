"""Tests for categorical cross-entropy loss."""

import unittest

import numpy as np

from training.error.categorial_cross_entropy import categorical_cross_entropy


class TestCategoricalCrossEntropy(unittest.TestCase):
    """Tests for categorical_cross_entropy."""

    def test_categorical_cross_entropy_returns_expected_key(self) -> None:
        """Return a dictionary containing the loss value."""
        y_one_hot = np.array([[1.0, 0.0, 0.0]])
        y_pred = np.array([[0.8, 0.1, 0.1]])

        result = categorical_cross_entropy(
            y_one_hot=y_one_hot,
            y_pred=y_pred,
        )

        self.assertEqual(set(result.keys()), {"loss"})

    def test_categorical_cross_entropy_returns_float_loss(self) -> None:
        """Return the loss value as a float."""
        y_one_hot = np.array([[1.0, 0.0, 0.0]])
        y_pred = np.array([[0.8, 0.1, 0.1]])

        result = categorical_cross_entropy(
            y_one_hot=y_one_hot,
            y_pred=y_pred,
        )

        self.assertIsInstance(result["loss"], float)

    def test_categorical_cross_entropy_computes_loss_for_single_example(self) -> None:
        """Compute cross-entropy loss for one example."""
        y_one_hot = np.array([[1.0, 0.0, 0.0]])
        y_pred = np.array([[0.8, 0.1, 0.1]])

        result = categorical_cross_entropy(
            y_one_hot=y_one_hot,
            y_pred=y_pred,
        )

        expected_loss = -np.log(0.8)

        self.assertAlmostEqual(result["loss"], expected_loss)

    def test_categorical_cross_entropy_computes_average_loss_for_multiple_examples(self) -> None:
        """Compute average cross-entropy loss across multiple examples."""
        y_one_hot = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        )
        y_pred = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.9],
            ],
        )

        result = categorical_cross_entropy(
            y_one_hot=y_one_hot,
            y_pred=y_pred,
        )

        expected_loss = -float((np.log(0.8) + np.log(0.7) + np.log(0.9)) / 3)

        self.assertAlmostEqual(result["loss"], expected_loss)

    def test_categorical_cross_entropy_returns_low_loss_for_confident_correct_predictions(
        self,
    ) -> None:
        """Return a near-zero loss when predictions are confidently correct."""
        y_one_hot = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
        )
        y_pred = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
        )

        result = categorical_cross_entropy(
            y_one_hot=y_one_hot,
            y_pred=y_pred,
        )

        self.assertGreaterEqual(result["loss"], 0.0)
        self.assertLess(result["loss"], 1e-12)

    def test_categorical_cross_entropy_clips_zero_probabilities(self) -> None:
        """Avoid infinite loss when the predicted probability is zero."""
        y_one_hot = np.array([[1.0, 0.0, 0.0]])
        y_pred = np.array([[0.0, 0.5, 0.5]])

        result = categorical_cross_entropy(
            y_one_hot=y_one_hot,
            y_pred=y_pred,
        )

        self.assertFalse(np.isnan(result["loss"]))
        self.assertFalse(np.isinf(result["loss"]))
        self.assertAlmostEqual(result["loss"], -np.log(1e-15))


if __name__ == "__main__":
    unittest.main()
