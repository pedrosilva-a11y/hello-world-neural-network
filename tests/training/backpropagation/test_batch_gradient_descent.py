"""Tests for batch gradient descent parameter updates."""

import unittest

import numpy as np

from training.backpropagation.batch_gradient_descent import batch_gradient_descent


class TestBatchGradientDescent(unittest.TestCase):
    """Tests for batch_gradient_descent."""

    def test_batch_gradient_descent_returns_expected_keys(self) -> None:
        """Return a dictionary containing updated W1 and b1."""
        W1 = np.array([[1.0, 2.0]])
        b1 = np.array([[0.5, -0.5]])
        dW1 = np.array([[0.1, 0.2]])
        db1 = np.array([[0.01, -0.01]])

        result = batch_gradient_descent(
            W1=W1,
            b1=b1,
            dW1=dW1,
            db1=db1,
        )

        self.assertEqual(set(result.keys()), {"W1", "b1"})

    def test_batch_gradient_descent_updates_weights(self) -> None:
        """Update W1 by subtracting learning rate times dW1."""
        W1 = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        b1 = np.array([[0.5, -0.5]])
        dW1 = np.array(
            [
                [0.1, 0.2],
                [0.3, 0.4],
            ],
        )
        db1 = np.array([[0.01, -0.01]])

        result = batch_gradient_descent(
            W1=W1,
            b1=b1,
            dW1=dW1,
            db1=db1,
            learning_rate=0.1,
        )

        expected_W1 = np.array(
            [
                [0.99, 1.98],
                [2.97, 3.96],
            ],
        )

        np.testing.assert_allclose(result["W1"], expected_W1)

    def test_batch_gradient_descent_updates_bias(self) -> None:
        """Update b1 by subtracting learning rate times db1."""
        W1 = np.array([[1.0, 2.0]])
        b1 = np.array([[0.5, -0.5]])
        dW1 = np.array([[0.1, 0.2]])
        db1 = np.array([[0.01, -0.01]])

        result = batch_gradient_descent(
            W1=W1,
            b1=b1,
            dW1=dW1,
            db1=db1,
            learning_rate=0.1,
        )

        expected_b1 = np.array([[0.499, -0.499]])

        np.testing.assert_allclose(result["b1"], expected_b1)

    def test_batch_gradient_descent_preserves_parameter_shapes(self) -> None:
        """Preserve W1 and b1 shapes after updating parameters."""
        W1 = np.zeros((784, 10))
        b1 = np.zeros((1, 10))
        dW1 = np.ones((784, 10))
        db1 = np.ones((1, 10))

        result = batch_gradient_descent(
            W1=W1,
            b1=b1,
            dW1=dW1,
            db1=db1,
        )

        self.assertEqual(result["W1"].shape, (784, 10))
        self.assertEqual(result["b1"].shape, (1, 10))

    def test_batch_gradient_descent_does_not_mutate_original_parameters(self) -> None:
        """Keep original W1 and b1 arrays unchanged."""
        W1 = np.array([[1.0, 2.0]])
        b1 = np.array([[0.5, -0.5]])
        dW1 = np.array([[0.1, 0.2]])
        db1 = np.array([[0.01, -0.01]])

        original_W1 = W1.copy()
        original_b1 = b1.copy()

        batch_gradient_descent(
            W1=W1,
            b1=b1,
            dW1=dW1,
            db1=db1,
            learning_rate=0.1,
        )

        np.testing.assert_array_equal(W1, original_W1)
        np.testing.assert_array_equal(b1, original_b1)


if __name__ == "__main__":
    unittest.main()
