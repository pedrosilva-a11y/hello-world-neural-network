"""Tests for batch gradient descent parameter updates."""

import unittest

import numpy as np

from training.backpropagation.optimizer.batch_gradient_descent import batch_gradient_descent


class TestBatchGradientDescent(unittest.TestCase):
    """Tests for batch_gradient_descent."""

    def test_batch_gradient_descent_returns_parameter_dictionary(self) -> None:
        """Return the updated parameter dictionary."""
        parameters = {
            "W1": np.array([[1.0, 2.0]]),
            "b1": np.array([[0.5, -0.5]]),
        }
        gradients = {
            "dW1": np.array([[0.1, 0.2]]),
            "db1": np.array([[0.01, -0.01]]),
        }

        result = batch_gradient_descent(
            gradients=gradients,
            parameters=parameters,
            layer=1,
        )

        self.assertEqual(set(result.keys()), {"W1", "b1"})

    def test_batch_gradient_descent_updates_weights_for_selected_layer(self) -> None:
        """Update W for the selected layer by subtracting learning rate times dW."""
        parameters = {
            "W1": np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
            ),
            "b1": np.array([[0.5, -0.5]]),
        }
        gradients = {
            "dW1": np.array(
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                ],
            ),
            "db1": np.array([[0.01, -0.01]]),
        }

        result = batch_gradient_descent(
            gradients=gradients,
            parameters=parameters,
            layer=1,
            learning_rate=0.1,
        )

        expected_W1 = np.array(
            [
                [0.99, 1.98],
                [2.97, 3.96],
            ],
        )

        np.testing.assert_allclose(result["W1"], expected_W1)

    def test_batch_gradient_descent_updates_bias_for_selected_layer(self) -> None:
        """Update b for the selected layer by subtracting learning rate times db."""
        parameters = {
            "W1": np.array([[1.0, 2.0]]),
            "b1": np.array([[0.5, -0.5]]),
        }
        gradients = {
            "dW1": np.array([[0.1, 0.2]]),
            "db1": np.array([[0.01, -0.01]]),
        }

        result = batch_gradient_descent(
            gradients=gradients,
            parameters=parameters,
            layer=1,
            learning_rate=0.1,
        )

        expected_b1 = np.array([[0.499, -0.499]])

        np.testing.assert_allclose(result["b1"], expected_b1)

    def test_batch_gradient_descent_updates_requested_layer_only(self) -> None:
        """Update only the parameters belonging to the requested layer."""
        parameters = {
            "W1": np.array([[1.0, 2.0]]),
            "b1": np.array([[0.5, -0.5]]),
            "W2": np.array([[3.0, 4.0]]),
            "b2": np.array([[1.0, -1.0]]),
        }
        gradients = {
            "dW2": np.array([[0.3, 0.4]]),
            "db2": np.array([[0.03, -0.04]]),
        }

        result = batch_gradient_descent(
            gradients=gradients,
            parameters=parameters,
            layer=2,
            learning_rate=0.1,
        )

        expected_W2 = np.array([[2.97, 3.96]])
        expected_b2 = np.array([[0.997, -0.996]])

        np.testing.assert_array_equal(result["W1"], np.array([[1.0, 2.0]]))
        np.testing.assert_array_equal(result["b1"], np.array([[0.5, -0.5]]))
        np.testing.assert_allclose(result["W2"], expected_W2)
        np.testing.assert_allclose(result["b2"], expected_b2)

    def test_batch_gradient_descent_preserves_parameter_shapes(self) -> None:
        """Preserve selected parameter shapes after updating them."""
        parameters = {
            "W1": np.zeros((784, 10)),
            "b1": np.zeros((1, 10)),
        }
        gradients = {
            "dW1": np.ones((784, 10)),
            "db1": np.ones((1, 10)),
        }

        result = batch_gradient_descent(
            gradients=gradients,
            parameters=parameters,
            layer=1,
        )

        self.assertEqual(result["W1"].shape, (784, 10))
        self.assertEqual(result["b1"].shape, (1, 10))

    def test_batch_gradient_descent_mutates_parameter_dictionary(self) -> None:
        """Mutate and return the same parameter dictionary object."""
        parameters = {
            "W1": np.array([[1.0, 2.0]]),
            "b1": np.array([[0.5, -0.5]]),
        }
        gradients = {
            "dW1": np.array([[0.1, 0.2]]),
            "db1": np.array([[0.01, -0.01]]),
        }

        result = batch_gradient_descent(
            gradients=gradients,
            parameters=parameters,
            layer=1,
            learning_rate=0.1,
        )

        self.assertIs(result, parameters)

        expected_W1 = np.array([[0.99, 1.98]])
        expected_b1 = np.array([[0.499, -0.499]])

        np.testing.assert_allclose(parameters["W1"], expected_W1)
        np.testing.assert_allclose(parameters["b1"], expected_b1)


if __name__ == "__main__":
    unittest.main()
