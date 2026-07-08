"""Tests for the softmax activation function."""

import unittest

import numpy as np

from training.activation.softmax import softmax


class TestSoftmax(unittest.TestCase):
    """Tests for softmax."""

    def test_softmax_returns_expected_key(self) -> None:
        """Return a dictionary containing the A activation matrix."""
        logits = np.array([[1.0, 2.0, 3.0]])

        result = softmax(logits=logits)

        self.assertEqual(set(result.keys()), {"A"})

    def test_softmax_preserves_input_shape(self) -> None:
        """Return probabilities with the same shape as the logits matrix."""
        logits = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
        )

        result = softmax(logits=logits)

        self.assertEqual(result["A"].shape, logits.shape)

    def test_softmax_rows_sum_to_one(self) -> None:
        """Return probabilities where each row sums to one."""
        logits = np.array(
            [
                [1.0, 2.0, 3.0],
                [10.0, 20.0, 30.0],
                [-1.0, 0.0, 1.0],
            ],
        )

        result = softmax(logits=logits)

        row_sums = np.sum(result["A"], axis=1)

        np.testing.assert_allclose(row_sums, np.array([1.0, 1.0, 1.0]))

    def test_softmax_returns_expected_probabilities_for_known_values(self) -> None:
        """Return expected softmax probabilities for a simple logits row."""
        logits = np.array([[1.0, 2.0, 3.0]])

        result = softmax(logits=logits)

        exp_logits = np.exp(np.array([[1.0, 2.0, 3.0]]) - 3.0)
        expected_probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        np.testing.assert_allclose(result["A"], expected_probabilities)

    def test_softmax_returns_uniform_probabilities_for_equal_logits(self) -> None:
        """Return equal probabilities when all logits in a row are equal."""
        logits = np.array([[5.0, 5.0, 5.0, 5.0]])

        result = softmax(logits=logits)

        expected_probabilities = np.array([[0.25, 0.25, 0.25, 0.25]])

        np.testing.assert_allclose(result["A"], expected_probabilities)

    def test_softmax_is_numerically_stable_for_large_logits(self) -> None:
        """Return finite probabilities for very large logits."""
        logits = np.array([[1000.0, 1001.0, 1002.0]])

        result = softmax(logits=logits)

        self.assertFalse(np.any(np.isnan(result["A"])))
        self.assertFalse(np.any(np.isinf(result["A"])))
        np.testing.assert_allclose(np.sum(result["A"], axis=1), np.array([1.0]))

    def test_softmax_is_invariant_to_adding_constant_to_logits(self) -> None:
        """Return the same probabilities when adding a constant to every logit."""
        logits = np.array([[1.0, 2.0, 3.0]])
        shifted_logits = logits + 100.0

        result = softmax(logits=logits)
        shifted_result = softmax(logits=shifted_logits)

        np.testing.assert_allclose(result["A"], shifted_result["A"])


if __name__ == "__main__":
    unittest.main()
