"""Tests for L2 weight decay regularization utilities."""

import unittest

import numpy as np

from training.regularization.weight_decay import (
    apply_weight_decay_to_gradients,
    weight_decay_gradient_term,
    weight_decay_loss_term,
)


class TestWeightDecay(unittest.TestCase):
    """Test L2 weight decay loss and gradient terms."""

    def test_weight_decay_loss_term_returns_zero_when_lambda_is_zero(self) -> None:
        """Return zero loss term when lambda is zero."""
        x_train = np.zeros((4, 2))
        parameters = {
            "W1": np.array([[1.0, -2.0], [3.0, 4.0]]),
            "b1": np.array([[10.0, 20.0]]),
        }

        result = weight_decay_loss_term(
            x_train=x_train,
            lambda_coefficient=0.0,
            parameters=parameters,
        )

        self.assertEqual(result, 0.0)

    def test_weight_decay_loss_term_uses_only_weight_parameters(self) -> None:
        """Compute the L2 loss term using weights and ignoring biases."""
        x_train = np.zeros((4, 2))
        parameters = {
            "W1": np.array([[1.0, -2.0], [3.0, 4.0]]),
            "b1": np.array([[100.0, 200.0]]),
            "W2": np.array([[-1.0], [2.0]]),
            "b2": np.array([[300.0]]),
        }

        result = weight_decay_loss_term(
            x_train=x_train,
            lambda_coefficient=0.2,
            parameters=parameters,
        )

        expected = 0.875

        self.assertAlmostEqual(result, expected)

    def test_weight_decay_loss_term_uses_x_train_shape_by_default(self) -> None:
        """Use x_train.shape[0] as the default L2 denominator."""
        x_train = np.zeros((4, 2))
        parameters = {
            "W1": np.array([[1.0, -2.0], [3.0, 4.0]]),
            "b1": np.array([[100.0, 200.0]]),
            "W2": np.array([[-1.0], [2.0]]),
            "b2": np.array([[300.0]]),
        }

        result = weight_decay_loss_term(
            x_train=x_train,
            lambda_coefficient=0.2,
            parameters=parameters,
        )

        weight_squared_sum = 35.0
        expected = (0.2 / (2 * 4)) * weight_squared_sum

        self.assertAlmostEqual(result, expected)

    def test_weight_decay_loss_term_uses_explicit_regularization_sample_count(
        self,
    ) -> None:
        """Use explicit full-training count instead of current batch size."""
        x_batch = np.zeros((2, 2))
        parameters = {
            "W1": np.array([[1.0, -2.0], [3.0, 4.0]]),
            "b1": np.array([[100.0, 200.0]]),
            "W2": np.array([[-1.0], [2.0]]),
            "b2": np.array([[300.0]]),
        }

        result = weight_decay_loss_term(
            x_train=x_batch,
            lambda_coefficient=0.2,
            parameters=parameters,
            regularization_sample_count=8,
        )

        weight_squared_sum = 35.0
        expected = (0.2 / (2 * 8)) * weight_squared_sum

        self.assertAlmostEqual(result, expected)

    def test_weight_decay_loss_term_explicit_count_differs_from_batch_size(
        self,
    ) -> None:
        """Ensure mini-batch L2 loss does not accidentally use batch size."""
        x_batch = np.zeros((2, 2))
        parameters = {
            "W1": np.array([[1.0, -2.0], [3.0, 4.0]]),
            "b1": np.array([[100.0, 200.0]]),
            "W2": np.array([[-1.0], [2.0]]),
            "b2": np.array([[300.0]]),
        }

        result = weight_decay_loss_term(
            x_train=x_batch,
            lambda_coefficient=0.2,
            parameters=parameters,
            regularization_sample_count=8,
        )

        weight_squared_sum = 35.0
        wrong_batch_scaled_result = (0.2 / (2 * 2)) * weight_squared_sum

        self.assertNotAlmostEqual(result, wrong_batch_scaled_result)

    def test_weight_decay_gradient_term_returns_zero_when_lambda_is_zero(self) -> None:
        """Return a zero gradient term when lambda is zero."""
        x_train = np.zeros((4, 2))
        weight_parameter = np.array([[1.0, -2.0], [3.0, 4.0]])

        result = weight_decay_gradient_term(
            x_train=x_train,
            lambda_coefficient=0.0,
            weight_parameter=weight_parameter,
        )

        np.testing.assert_array_equal(result, np.zeros_like(weight_parameter))

    def test_weight_decay_gradient_term_scales_weight_parameter(self) -> None:
        """Scale the weight matrix by lambda divided by sample count."""
        x_train = np.zeros((4, 2))
        weight_parameter = np.array([[1.0, -2.0], [3.0, 4.0]])

        result = weight_decay_gradient_term(
            x_train=x_train,
            lambda_coefficient=0.2,
            weight_parameter=weight_parameter,
        )

        expected = np.array([[0.05, -0.1], [0.15, 0.2]])

        np.testing.assert_allclose(result, expected)

    def test_weight_decay_gradient_term_uses_x_train_shape_by_default(self) -> None:
        """Use x_train.shape[0] as the default L2 gradient denominator."""
        x_train = np.zeros((4, 2))
        weight_parameter = np.array([[1.0, -2.0], [3.0, 4.0]])

        result = weight_decay_gradient_term(
            x_train=x_train,
            lambda_coefficient=0.2,
            weight_parameter=weight_parameter,
        )

        expected = (0.2 / 4) * weight_parameter

        np.testing.assert_allclose(result, expected)

    def test_weight_decay_gradient_term_uses_explicit_regularization_sample_count(
        self,
    ) -> None:
        """Use explicit full-training count instead of current batch size."""
        x_batch = np.zeros((2, 2))
        weight_parameter = np.array([[1.0, -2.0], [3.0, 4.0]])

        result = weight_decay_gradient_term(
            x_train=x_batch,
            lambda_coefficient=0.2,
            weight_parameter=weight_parameter,
            regularization_sample_count=8,
        )

        expected = (0.2 / 8) * weight_parameter

        np.testing.assert_allclose(result, expected)

    def test_weight_decay_gradient_term_explicit_count_differs_from_batch_size(
        self,
    ) -> None:
        """Ensure mini-batch L2 gradient does not accidentally use batch size."""
        x_batch = np.zeros((2, 2))
        weight_parameter = np.array([[1.0, -2.0], [3.0, 4.0]])

        result = weight_decay_gradient_term(
            x_train=x_batch,
            lambda_coefficient=0.2,
            weight_parameter=weight_parameter,
            regularization_sample_count=8,
        )

        wrong_batch_scaled_result = (0.2 / 2) * weight_parameter

        self.assertFalse(np.allclose(result, wrong_batch_scaled_result))

    def test_apply_weight_decay_to_gradients_updates_only_weight_gradients(self) -> None:
        """Add L2 terms to dW gradients and leave db gradients unchanged."""
        x_train = np.zeros((4, 2))
        gradients = {
            "dW1": np.ones((2, 2)),
            "db1": np.array([[1.0, 1.0]]),
            "dW2": np.full((2, 1), 2.0),
            "db2": np.array([[2.0]]),
        }
        parameters = {
            "W1": np.array([[1.0, -2.0], [3.0, 4.0]]),
            "b1": np.array([[100.0, 200.0]]),
            "W2": np.array([[-1.0], [2.0]]),
            "b2": np.array([[300.0]]),
        }

        result = apply_weight_decay_to_gradients(
            x_train=x_train,
            lambda_coefficient=0.2,
            gradients=gradients,
            parameters=parameters,
            layers=2,
        )

        expected_dW1 = np.array([[1.05, 0.9], [1.15, 1.2]])
        expected_dW2 = np.array([[1.95], [2.1]])

        np.testing.assert_allclose(result["dW1"], expected_dW1)
        np.testing.assert_allclose(result["dW2"], expected_dW2)
        np.testing.assert_array_equal(result["db1"], np.array([[1.0, 1.0]]))
        np.testing.assert_array_equal(result["db2"], np.array([[2.0]]))

    def test_apply_weight_decay_to_gradients_uses_explicit_regularization_sample_count(
        self,
    ) -> None:
        """Apply L2 gradients using explicit full-training denominator."""
        x_batch = np.zeros((2, 2))
        gradients = {
            "dW1": np.ones((2, 2)),
            "db1": np.array([[1.0, 1.0]]),
            "dW2": np.full((2, 1), 2.0),
            "db2": np.array([[2.0]]),
        }
        parameters = {
            "W1": np.array([[1.0, -2.0], [3.0, 4.0]]),
            "b1": np.array([[100.0, 200.0]]),
            "W2": np.array([[-1.0], [2.0]]),
            "b2": np.array([[300.0]]),
        }

        result = apply_weight_decay_to_gradients(
            x_train=x_batch,
            lambda_coefficient=0.2,
            gradients=gradients,
            parameters=parameters,
            layers=2,
            regularization_sample_count=8,
        )

        expected_dW1 = np.ones((2, 2)) + (0.2 / 8) * parameters["W1"]
        expected_dW2 = np.full((2, 1), 2.0) + (0.2 / 8) * parameters["W2"]

        np.testing.assert_allclose(result["dW1"], expected_dW1)
        np.testing.assert_allclose(result["dW2"], expected_dW2)
        np.testing.assert_array_equal(result["db1"], np.array([[1.0, 1.0]]))
        np.testing.assert_array_equal(result["db2"], np.array([[2.0]]))

    def test_apply_weight_decay_to_gradients_returns_same_gradients_when_lambda_is_zero(
        self,
    ) -> None:
        """Do not modify gradients when lambda is zero."""
        x_train = np.zeros((4, 2))
        gradients = {
            "dW1": np.ones((2, 2)),
            "db1": np.array([[1.0, 1.0]]),
        }
        parameters = {
            "W1": np.array([[1.0, -2.0], [3.0, 4.0]]),
            "b1": np.array([[100.0, 200.0]]),
        }

        result = apply_weight_decay_to_gradients(
            x_train=x_train,
            lambda_coefficient=0.0,
            gradients=gradients,
            parameters=parameters,
            layers=1,
            regularization_sample_count=8,
        )

        np.testing.assert_array_equal(result["dW1"], np.ones((2, 2)))
        np.testing.assert_array_equal(result["db1"], np.array([[1.0, 1.0]]))

    def test_weight_decay_loss_term_rejects_zero_regularization_sample_count(
        self,
    ) -> None:
        """Reject zero explicit regularization denominator."""
        x_train = np.zeros((4, 2))
        parameters = {
            "W1": np.array([[1.0, -2.0], [3.0, 4.0]]),
        }

        with self.assertRaisesRegex(
            ValueError,
            "regularization_sample_count must be at least 1",
        ):
            weight_decay_loss_term(
                x_train=x_train,
                lambda_coefficient=0.2,
                parameters=parameters,
                regularization_sample_count=0,
            )

    def test_weight_decay_gradient_term_rejects_zero_regularization_sample_count(
        self,
    ) -> None:
        """Reject zero explicit regularization denominator."""
        x_train = np.zeros((4, 2))
        weight_parameter = np.array([[1.0, -2.0], [3.0, 4.0]])

        with self.assertRaisesRegex(
            ValueError,
            "regularization_sample_count must be at least 1",
        ):
            weight_decay_gradient_term(
                x_train=x_train,
                lambda_coefficient=0.2,
                weight_parameter=weight_parameter,
                regularization_sample_count=0,
            )

    def test_weight_decay_gradient_term_rejects_negative_regularization_sample_count(
        self,
    ) -> None:
        """Reject negative explicit regularization denominator."""
        x_train = np.zeros((4, 2))
        weight_parameter = np.array([[1.0, -2.0], [3.0, 4.0]])

        with self.assertRaisesRegex(
            ValueError,
            "regularization_sample_count must be at least 1",
        ):
            weight_decay_gradient_term(
                x_train=x_train,
                lambda_coefficient=0.2,
                weight_parameter=weight_parameter,
                regularization_sample_count=-1,
            )


if __name__ == "__main__":
    unittest.main()
