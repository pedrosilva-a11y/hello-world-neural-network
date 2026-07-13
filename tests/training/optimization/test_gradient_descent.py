"""Tests for gradient descent optimizer utilities."""

import unittest

import numpy as np

from training.optimization.gradient_descent import (
    update_parameters_with_gradient_descent,
)


class TestUpdateParametersWithGradientDescent(unittest.TestCase):
    """Tests for update_parameters_with_gradient_descent."""

    def test_update_parameters_with_gradient_descent_updates_weights_and_biases(
        self,
    ) -> None:
        """Update each parameter using its matching gradient."""
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
                    [0.1, -0.2],
                    [0.3, -0.4],
                ],
            ),
            "db1": np.array([[0.05, -0.05]]),
        }

        result = update_parameters_with_gradient_descent(
            parameters=parameters,
            gradients=gradients,
            learning_rate=0.1,
        )

        expected_w1 = np.array(
            [
                [0.99, 2.02],
                [2.97, 4.04],
            ],
        )
        expected_b1 = np.array([[0.495, -0.495]])

        np.testing.assert_allclose(result["W1"], expected_w1)
        np.testing.assert_allclose(result["b1"], expected_b1)

    def test_update_parameters_with_gradient_descent_does_not_mutate_inputs(
        self,
    ) -> None:
        """Return updated parameters without mutating input parameters."""
        parameters = {
            "W1": np.array([[1.0, 2.0]]),
            "b1": np.array([[0.5, -0.5]]),
        }
        gradients = {
            "dW1": np.array([[0.2, -0.4]]),
            "db1": np.array([[0.1, -0.1]]),
        }

        original_w1 = parameters["W1"].copy()
        original_b1 = parameters["b1"].copy()

        result = update_parameters_with_gradient_descent(
            parameters=parameters,
            gradients=gradients,
            learning_rate=0.1,
        )

        np.testing.assert_array_equal(parameters["W1"], original_w1)
        np.testing.assert_array_equal(parameters["b1"], original_b1)

        self.assertIsNot(result["W1"], parameters["W1"])
        self.assertIsNot(result["b1"], parameters["b1"])

    def test_update_parameters_with_gradient_descent_raises_error_for_missing_gradient(
        self,
    ) -> None:
        """Raise KeyError when a parameter gradient is missing."""
        parameters = {
            "W1": np.array([[1.0, 2.0]]),
            "b1": np.array([[0.5, -0.5]]),
        }
        gradients = {
            "dW1": np.array([[0.2, -0.4]]),
        }

        with self.assertRaisesRegex(
            KeyError,
            "Missing gradient for parameter: b1",
        ):
            update_parameters_with_gradient_descent(
                parameters=parameters,
                gradients=gradients,
                learning_rate=0.1,
            )

    def test_update_parameters_with_gradient_descent_supports_multiple_layers(
        self,
    ) -> None:
        """Update parameters across multiple layers."""
        parameters = {
            "W1": np.array([[1.0, 2.0]]),
            "b1": np.array([[0.5, -0.5]]),
            "W2": np.array(
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                ],
            ),
            "b2": np.array([[0.0, 0.1]]),
        }
        gradients = {
            "dW1": np.array([[0.1, 0.2]]),
            "db1": np.array([[0.05, -0.05]]),
            "dW2": np.array(
                [
                    [0.01, 0.02],
                    [0.03, 0.04],
                ],
            ),
            "db2": np.array([[0.1, -0.1]]),
        }

        result = update_parameters_with_gradient_descent(
            parameters=parameters,
            gradients=gradients,
            learning_rate=0.5,
        )

        np.testing.assert_allclose(result["W1"], np.array([[0.95, 1.9]]))
        np.testing.assert_allclose(result["b1"], np.array([[0.475, -0.475]]))
        np.testing.assert_allclose(
            result["W2"],
            np.array(
                [
                    [0.095, 0.19],
                    [0.285, 0.38],
                ],
            ),
        )
        np.testing.assert_allclose(result["b2"], np.array([[-0.05, 0.15]]))


if __name__ == "__main__":
    unittest.main()
