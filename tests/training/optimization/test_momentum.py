"""Tests for momentum optimizer utilities."""

import unittest

import numpy as np

from training.optimization.momentum import (
    initialize_velocity,
    update_parameters_with_momentum,
)


class TestInitializeVelocity(unittest.TestCase):
    """Tests for initialize_velocity."""

    def test_initialize_velocity_returns_zero_arrays_matching_parameter_shapes(
        self,
    ) -> None:
        """Initialize one zero velocity array for each parameter."""
        parameters = {
            "W1": np.array(
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                ],
            ),
            "b1": np.array([[0.0, 0.0]]),
            "W2": np.array(
                [
                    [0.5, 0.6],
                    [0.7, 0.8],
                ],
            ),
            "b2": np.array([[0.1, -0.1]]),
        }

        result = initialize_velocity(parameters=parameters)

        self.assertEqual(set(result.keys()), {"W1", "b1", "W2", "b2"})

        np.testing.assert_array_equal(result["W1"], np.zeros((2, 2)))
        np.testing.assert_array_equal(result["b1"], np.zeros((1, 2)))
        np.testing.assert_array_equal(result["W2"], np.zeros((2, 2)))
        np.testing.assert_array_equal(result["b2"], np.zeros((1, 2)))

    def test_initialize_velocity_does_not_reuse_parameter_array_objects(
        self,
    ) -> None:
        """Return velocity arrays that are independent from parameter arrays."""
        parameters = {
            "W1": np.array([[0.1, 0.2]]),
            "b1": np.array([[0.0, 0.0]]),
        }

        result = initialize_velocity(parameters=parameters)

        self.assertIsNot(result["W1"], parameters["W1"])
        self.assertIsNot(result["b1"], parameters["b1"])


class TestUpdateParametersWithMomentum(unittest.TestCase):
    """Tests for update_parameters_with_momentum."""

    def test_update_parameters_with_momentum_updates_parameters_and_velocity(
        self,
    ) -> None:
        """Update parameters using beta-scaled velocity and current gradients."""
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
        velocity = {
            "W1": np.array(
                [
                    [0.5, 0.5],
                    [0.5, 0.5],
                ],
            ),
            "b1": np.array([[0.1, 0.1]]),
        }

        updated_parameters, updated_velocity = update_parameters_with_momentum(
            parameters=parameters,
            gradients=gradients,
            velocity=velocity,
            learning_rate=0.1,
            beta=0.9,
        )

        expected_w1_velocity = np.array(
            [
                [0.55, 0.25],
                [0.75, 0.05],
            ],
        )
        expected_b1_velocity = np.array([[0.14, 0.04]])

        expected_w1_parameters = np.array(
            [
                [0.945, 1.975],
                [2.925, 3.995],
            ],
        )
        expected_b1_parameters = np.array([[0.486, -0.504]])

        np.testing.assert_allclose(updated_velocity["W1"], expected_w1_velocity)
        np.testing.assert_allclose(updated_velocity["b1"], expected_b1_velocity)
        np.testing.assert_allclose(updated_parameters["W1"], expected_w1_parameters)
        np.testing.assert_allclose(updated_parameters["b1"], expected_b1_parameters)

    def test_update_parameters_with_momentum_matches_gradient_descent_when_beta_is_zero(
        self,
    ) -> None:
        """Use only the current gradient when beta is zero."""
        parameters = {
            "W1": np.array([[1.0, 2.0]]),
            "b1": np.array([[0.5, -0.5]]),
        }
        gradients = {
            "dW1": np.array([[0.2, -0.4]]),
            "db1": np.array([[0.1, -0.1]]),
        }
        velocity = {
            "W1": np.array([[10.0, 10.0]]),
            "b1": np.array([[10.0, 10.0]]),
        }

        updated_parameters, updated_velocity = update_parameters_with_momentum(
            parameters=parameters,
            gradients=gradients,
            velocity=velocity,
            learning_rate=0.1,
            beta=0.0,
        )

        expected_w1_parameters = np.array([[0.98, 2.04]])
        expected_b1_parameters = np.array([[0.49, -0.49]])

        np.testing.assert_allclose(updated_velocity["W1"], gradients["dW1"])
        np.testing.assert_allclose(updated_velocity["b1"], gradients["db1"])
        np.testing.assert_allclose(updated_parameters["W1"], expected_w1_parameters)
        np.testing.assert_allclose(updated_parameters["b1"], expected_b1_parameters)

    def test_update_parameters_with_momentum_accumulates_velocity_across_steps(
        self,
    ) -> None:
        """Use previous velocity when computing the next update."""
        parameters = {
            "W1": np.array([[1.0, 2.0]]),
            "b1": np.array([[0.0, 0.0]]),
        }
        velocity = initialize_velocity(parameters=parameters)

        first_gradients = {
            "dW1": np.array([[0.2, -0.2]]),
            "db1": np.array([[0.1, -0.1]]),
        }

        first_parameters, first_velocity = update_parameters_with_momentum(
            parameters=parameters,
            gradients=first_gradients,
            velocity=velocity,
            learning_rate=0.1,
            beta=0.9,
        )

        second_gradients = {
            "dW1": np.array([[0.2, -0.2]]),
            "db1": np.array([[0.1, -0.1]]),
        }

        second_parameters, second_velocity = update_parameters_with_momentum(
            parameters=first_parameters,
            gradients=second_gradients,
            velocity=first_velocity,
            learning_rate=0.1,
            beta=0.9,
        )

        expected_second_w1_velocity = np.array([[0.38, -0.38]])
        expected_second_b1_velocity = np.array([[0.19, -0.19]])

        expected_second_w1_parameters = np.array([[0.942, 2.058]])
        expected_second_b1_parameters = np.array([[-0.029, 0.029]])

        np.testing.assert_allclose(second_velocity["W1"], expected_second_w1_velocity)
        np.testing.assert_allclose(second_velocity["b1"], expected_second_b1_velocity)
        np.testing.assert_allclose(second_parameters["W1"], expected_second_w1_parameters)
        np.testing.assert_allclose(second_parameters["b1"], expected_second_b1_parameters)

    def test_update_parameters_with_momentum_does_not_mutate_inputs(self) -> None:
        """Return new dictionaries without mutating parameters or velocity."""
        parameters = {
            "W1": np.array([[1.0, 2.0]]),
            "b1": np.array([[0.5, -0.5]]),
        }
        gradients = {
            "dW1": np.array([[0.2, -0.4]]),
            "db1": np.array([[0.1, -0.1]]),
        }
        velocity = {
            "W1": np.array([[0.3, 0.3]]),
            "b1": np.array([[0.2, 0.2]]),
        }

        original_w1 = parameters["W1"].copy()
        original_b1 = parameters["b1"].copy()
        original_w1_velocity = velocity["W1"].copy()
        original_b1_velocity = velocity["b1"].copy()

        update_parameters_with_momentum(
            parameters=parameters,
            gradients=gradients,
            velocity=velocity,
            learning_rate=0.1,
            beta=0.9,
        )

        np.testing.assert_array_equal(parameters["W1"], original_w1)
        np.testing.assert_array_equal(parameters["b1"], original_b1)
        np.testing.assert_array_equal(velocity["W1"], original_w1_velocity)
        np.testing.assert_array_equal(velocity["b1"], original_b1_velocity)

    def test_update_parameters_with_momentum_raises_error_for_negative_beta(
        self,
    ) -> None:
        """Raise ValueError when beta is negative."""
        parameters = {"W1": np.array([[1.0]])}
        gradients = {"dW1": np.array([[0.1]])}
        velocity = {"W1": np.array([[0.0]])}

        with self.assertRaisesRegex(
            ValueError,
            "beta must be greater than or equal to 0.0 and less than 1.0.",
        ):
            update_parameters_with_momentum(
                parameters=parameters,
                gradients=gradients,
                velocity=velocity,
                learning_rate=0.1,
                beta=-0.1,
            )

    def test_update_parameters_with_momentum_raises_error_for_beta_equal_to_one(
        self,
    ) -> None:
        """Raise ValueError when beta is equal to one."""
        parameters = {"W1": np.array([[1.0]])}
        gradients = {"dW1": np.array([[0.1]])}
        velocity = {"W1": np.array([[0.0]])}

        with self.assertRaisesRegex(
            ValueError,
            "beta must be greater than or equal to 0.0 and less than 1.0.",
        ):
            update_parameters_with_momentum(
                parameters=parameters,
                gradients=gradients,
                velocity=velocity,
                learning_rate=0.1,
                beta=1.0,
            )

    def test_update_parameters_with_momentum_raises_error_for_missing_gradient(
        self,
    ) -> None:
        """Raise KeyError when a parameter gradient is missing."""
        parameters = {"W1": np.array([[1.0]])}
        gradients: dict[str, np.ndarray] = {}
        velocity = {"W1": np.array([[0.0]])}

        with self.assertRaisesRegex(
            KeyError,
            "Missing gradient for parameter: W1",
        ):
            update_parameters_with_momentum(
                parameters=parameters,
                gradients=gradients,
                velocity=velocity,
                learning_rate=0.1,
                beta=0.9,
            )

    def test_update_parameters_with_momentum_raises_error_for_missing_velocity(
        self,
    ) -> None:
        """Raise KeyError when a parameter velocity is missing."""
        parameters = {"W1": np.array([[1.0]])}
        gradients = {"dW1": np.array([[0.1]])}
        velocity: dict[str, np.ndarray] = {}

        with self.assertRaisesRegex(
            KeyError,
            "Missing velocity for parameter: W1",
        ):
            update_parameters_with_momentum(
                parameters=parameters,
                gradients=gradients,
                velocity=velocity,
                learning_rate=0.1,
                beta=0.9,
            )


if __name__ == "__main__":
    unittest.main()
