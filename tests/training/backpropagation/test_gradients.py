"""Tests for softmax backpropagation gradient computations."""

import unittest

import numpy as np

from training.backpropagation.gradients import gradient_computations_softmax


class TestGradientComputationsSoftmax(unittest.TestCase):
    """Tests for gradient_computations_softmax."""

    def test_gradient_computations_softmax_returns_expected_keys(self) -> None:
        """Return a dictionary containing dZ, dW, and db."""
        x = np.array([[1.0, 2.0]])
        yhot = np.array([[1.0, 0.0]])
        activation = np.array([[0.8, 0.2]])

        result = gradient_computations_softmax(
            x=x,
            yhot=yhot,
            activation=activation,
        )

        self.assertEqual(set(result.keys()), {"dZ", "dW", "db"})

    def test_gradient_computations_softmax_computes_dz(self) -> None:
        """Compute dZ as activation minus one-hot labels."""
        x = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        yhot = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )
        activation = np.array(
            [
                [0.8, 0.2],
                [0.3, 0.7],
            ],
        )

        result = gradient_computations_softmax(
            x=x,
            yhot=yhot,
            activation=activation,
        )

        expected_dz = np.array(
            [
                [-0.2, 0.2],
                [0.3, -0.3],
            ],
        )

        np.testing.assert_allclose(result["dZ"], expected_dz)

    def test_gradient_computations_softmax_computes_dw(self) -> None:
        """Compute dW as X transpose multiplied by dZ, averaged by examples."""
        x = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        yhot = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )
        activation = np.array(
            [
                [0.8, 0.2],
                [0.3, 0.7],
            ],
        )

        result = gradient_computations_softmax(
            x=x,
            yhot=yhot,
            activation=activation,
        )

        expected_dw = np.array(
            [
                [0.35, -0.35],
                [0.4, -0.4],
            ],
        )

        np.testing.assert_allclose(result["dW"], expected_dw)

    def test_gradient_computations_softmax_computes_db(self) -> None:
        """Compute db as the average dZ value per output class."""
        x = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        yhot = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )
        activation = np.array(
            [
                [0.8, 0.2],
                [0.3, 0.7],
            ],
        )

        result = gradient_computations_softmax(
            x=x,
            yhot=yhot,
            activation=activation,
        )

        expected_db = np.array([[0.05, -0.05]])

        np.testing.assert_allclose(result["db"], expected_db)

    def test_gradient_computations_softmax_returns_expected_shapes(self) -> None:
        """Return gradients with shapes matching Z, W, and b."""
        x = np.zeros((5, 3))
        yhot = np.zeros((5, 4))
        activation = np.zeros((5, 4))

        result = gradient_computations_softmax(
            x=x,
            yhot=yhot,
            activation=activation,
        )

        self.assertEqual(result["dZ"].shape, (5, 4))
        self.assertEqual(result["dW"].shape, (3, 4))
        self.assertEqual(result["db"].shape, (1, 4))


if __name__ == "__main__":
    unittest.main()
