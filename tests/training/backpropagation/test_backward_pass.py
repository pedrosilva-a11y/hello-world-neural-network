"""Tests for the Digit Recognizer backward pass."""

import unittest
from unittest.mock import patch

import numpy as np

from training.backpropagation import backward_pass


class TestRunBackwardPass(unittest.TestCase):
    """Tests for run_backward_pass."""

    def test_run_backward_pass_coordinates_single_layer_backward_computations(
        self,
    ) -> None:
        """Run loss, softmax gradients, and weight-decay gradient computations."""
        x_train = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        y_one_hot = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )
        a1 = np.array(
            [
                [0.8, 0.2],
                [0.3, 0.7],
            ],
        )
        forward_pass_results = {
            "Y_one_hot": y_one_hot,
            "A1": a1,
        }
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
            "dZ1": np.array(
                [
                    [-0.2, 0.2],
                    [0.3, -0.3],
                ],
            ),
            "dW1": np.array(
                [
                    [0.35, -0.35],
                    [0.4, -0.4],
                ],
            ),
            "db1": np.array([[0.05, -0.05]]),
        }

        with (
            patch.object(
                backward_pass,
                "categorical_cross_entropy",
                return_value=0.289909,
            ) as mock_categorical_cross_entropy,
            patch.object(
                backward_pass,
                "weight_decay_loss_term",
                return_value=0.2,
            ) as mock_weight_decay_loss_term,
            patch.object(
                backward_pass,
                "gradient_computations_softmax",
                return_value=gradients,
            ) as mock_gradient_computations_softmax,
            patch.object(
                backward_pass,
                "gradient_computations_relu",
            ) as mock_gradient_computations_relu,
            patch.object(
                backward_pass,
                "apply_weight_decay_to_gradients",
                return_value=gradients,
            ) as mock_apply_weight_decay_to_gradients,
        ):
            result = backward_pass.run_backward_pass(
                x_train=x_train,
                forward_pass_results=forward_pass_results,
                parameters=parameters,
                neurons_profile=[2],
                lambda_coefficient=0.3,
            )

        self.assertAlmostEqual(result["loss"], 0.489909)
        self.assertIs(result["gradients"], gradients)
        self.assertNotIn("parameters", result)

        mock_categorical_cross_entropy.assert_called_once_with(
            y_one_hot=y_one_hot,
            y_pred=a1,
        )

        weight_decay_loss_kwargs = mock_weight_decay_loss_term.call_args.kwargs
        self.assertIs(weight_decay_loss_kwargs["x_train"], x_train)
        self.assertEqual(weight_decay_loss_kwargs["lambda_coefficient"], 0.3)
        self.assertIs(weight_decay_loss_kwargs["parameters"], parameters)
        self.assertIsNone(
            weight_decay_loss_kwargs["regularization_sample_count"],
        )

        mock_gradient_computations_softmax.assert_called_once_with(
            x=x_train,
            yhot=y_one_hot,
            forward_pass_results=forward_pass_results,
            layer=1,
        )
        mock_gradient_computations_relu.assert_not_called()

        weight_decay_gradient_kwargs = (
            mock_apply_weight_decay_to_gradients.call_args.kwargs
        )
        self.assertIs(weight_decay_gradient_kwargs["x_train"], x_train)
        self.assertEqual(weight_decay_gradient_kwargs["lambda_coefficient"], 0.3)
        self.assertIs(weight_decay_gradient_kwargs["gradients"], gradients)
        self.assertIs(weight_decay_gradient_kwargs["parameters"], parameters)
        self.assertEqual(weight_decay_gradient_kwargs["layers"], 1)
        self.assertIsNone(
            weight_decay_gradient_kwargs["regularization_sample_count"],
        )

    def test_run_backward_pass_forwards_regularization_sample_count(
        self,
    ) -> None:
        """Forward explicit L2 denominator to loss and gradient helpers."""
        x_batch = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        y_one_hot = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )
        a1 = np.array(
            [
                [0.8, 0.2],
                [0.3, 0.7],
            ],
        )
        forward_pass_results = {
            "Y_one_hot": y_one_hot,
            "A1": a1,
        }
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
            "dZ1": np.array(
                [
                    [-0.2, 0.2],
                    [0.3, -0.3],
                ],
            ),
            "dW1": np.array(
                [
                    [0.35, -0.35],
                    [0.4, -0.4],
                ],
            ),
            "db1": np.array([[0.05, -0.05]]),
        }

        with (
            patch.object(
                backward_pass,
                "categorical_cross_entropy",
                return_value=0.289909,
            ),
            patch.object(
                backward_pass,
                "weight_decay_loss_term",
                return_value=0.2,
            ) as mock_weight_decay_loss_term,
            patch.object(
                backward_pass,
                "gradient_computations_softmax",
                return_value=gradients,
            ),
            patch.object(
                backward_pass,
                "gradient_computations_relu",
            ),
            patch.object(
                backward_pass,
                "apply_weight_decay_to_gradients",
                return_value=gradients,
            ) as mock_apply_weight_decay_to_gradients,
        ):
            result = backward_pass.run_backward_pass(
                x_train=x_batch,
                forward_pass_results=forward_pass_results,
                parameters=parameters,
                neurons_profile=[2],
                lambda_coefficient=0.3,
                regularization_sample_count=8,
            )

        self.assertAlmostEqual(result["loss"], 0.489909)
        self.assertIs(result["gradients"], gradients)
        self.assertNotIn("parameters", result)

        weight_decay_loss_kwargs = mock_weight_decay_loss_term.call_args.kwargs
        self.assertIs(weight_decay_loss_kwargs["x_train"], x_batch)
        self.assertEqual(weight_decay_loss_kwargs["lambda_coefficient"], 0.3)
        self.assertIs(weight_decay_loss_kwargs["parameters"], parameters)
        self.assertEqual(
            weight_decay_loss_kwargs["regularization_sample_count"],
            8,
        )

        weight_decay_gradient_kwargs = (
            mock_apply_weight_decay_to_gradients.call_args.kwargs
        )
        self.assertIs(weight_decay_gradient_kwargs["x_train"], x_batch)
        self.assertEqual(weight_decay_gradient_kwargs["lambda_coefficient"], 0.3)
        self.assertIs(weight_decay_gradient_kwargs["gradients"], gradients)
        self.assertIs(weight_decay_gradient_kwargs["parameters"], parameters)
        self.assertEqual(weight_decay_gradient_kwargs["layers"], 1)
        self.assertEqual(
            weight_decay_gradient_kwargs["regularization_sample_count"],
            8,
        )

    def test_run_backward_pass_coordinates_hidden_layer_backward_computations(
        self,
    ) -> None:
        """Compute gradients backward before applying weight decay."""
        x_train = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        y_one_hot = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )
        forward_pass_results = {
            "Y_one_hot": y_one_hot,
            "Z1": np.array(
                [
                    [1.0, -1.0, 0.0],
                    [-2.0, 3.0, 4.0],
                ],
            ),
            "A1": np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 3.0, 4.0],
                ],
            ),
            "A2": np.array(
                [
                    [0.8, 0.2],
                    [0.3, 0.7],
                ],
            ),
        }
        parameters = {
            "W1": np.zeros((2, 3)),
            "b1": np.zeros((1, 3)),
            "W2": np.zeros((3, 2)),
            "b2": np.zeros((1, 2)),
        }

        softmax_gradients = {
            "dZ2": np.array(
                [
                    [-0.2, 0.2],
                    [0.3, -0.3],
                ],
            ),
            "dW2": np.zeros((3, 2)),
            "db2": np.zeros((1, 2)),
        }
        full_gradients = {
            **softmax_gradients,
            "dA1": np.zeros((2, 3)),
            "dZ1": np.zeros((2, 3)),
            "dW1": np.zeros((2, 3)),
            "db1": np.zeros((1, 3)),
        }

        with (
            patch.object(
                backward_pass,
                "categorical_cross_entropy",
                return_value=0.5,
            ),
            patch.object(
                backward_pass,
                "weight_decay_loss_term",
                return_value=0.1,
            ) as mock_weight_decay_loss_term,
            patch.object(
                backward_pass,
                "gradient_computations_softmax",
                return_value=softmax_gradients,
            ) as mock_gradient_computations_softmax,
            patch.object(
                backward_pass,
                "gradient_computations_relu",
                return_value=full_gradients,
            ) as mock_gradient_computations_relu,
            patch.object(
                backward_pass,
                "apply_weight_decay_to_gradients",
                return_value=full_gradients,
            ) as mock_apply_weight_decay_to_gradients,
        ):
            result = backward_pass.run_backward_pass(
                x_train=x_train,
                forward_pass_results=forward_pass_results,
                parameters=parameters,
                neurons_profile=[3, 2],
                lambda_coefficient=0.4,
                regularization_sample_count=8,
            )

        self.assertAlmostEqual(result["loss"], 0.6)
        self.assertIs(result["gradients"], full_gradients)
        self.assertNotIn("parameters", result)

        mock_weight_decay_loss_term.assert_called_once()
        self.assertEqual(
            mock_weight_decay_loss_term.call_args.kwargs["lambda_coefficient"],
            0.4,
        )
        self.assertEqual(
            mock_weight_decay_loss_term.call_args.kwargs[
                "regularization_sample_count"
            ],
            8,
        )

        mock_gradient_computations_softmax.assert_called_once_with(
            x=x_train,
            yhot=y_one_hot,
            forward_pass_results=forward_pass_results,
            layer=2,
        )
        mock_gradient_computations_relu.assert_called_once_with(
            x=x_train,
            gradients=softmax_gradients,
            parameters=parameters,
            forward_pass_results=forward_pass_results,
            layer=1,
        )

        mock_apply_weight_decay_to_gradients.assert_called_once()
        self.assertEqual(
            mock_apply_weight_decay_to_gradients.call_args.kwargs["layers"],
            2,
        )
        self.assertEqual(
            mock_apply_weight_decay_to_gradients.call_args.kwargs[
                "lambda_coefficient"
            ],
            0.4,
        )
        self.assertEqual(
            mock_apply_weight_decay_to_gradients.call_args.kwargs[
                "regularization_sample_count"
            ],
            8,
        )

    def test_run_backward_pass_returns_expected_shapes_for_single_layer(self) -> None:
        """Return loss and gradients for a softmax model."""
        x_train = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        y_one_hot = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )
        forward_pass_results = {
            "Y_one_hot": y_one_hot,
            "A1": np.array(
                [
                    [0.8, 0.2],
                    [0.3, 0.7],
                ],
            ),
        }
        parameters = {
            "W1": np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
            ),
            "b1": np.array([[0.5, -0.5]]),
        }

        result = backward_pass.run_backward_pass(
            x_train=x_train,
            forward_pass_results=forward_pass_results,
            parameters=parameters,
            neurons_profile=[2],
            lambda_coefficient=0.0,
        )

        self.assertIsInstance(result["loss"], float)
        self.assertEqual(result["gradients"]["dZ1"].shape, (2, 2))
        self.assertEqual(result["gradients"]["dW1"].shape, (2, 2))
        self.assertEqual(result["gradients"]["db1"].shape, (1, 2))
        self.assertNotIn("parameters", result)

    def test_run_backward_pass_returns_single_layer_gradients_without_parameter_update(
        self,
    ) -> None:
        """Return gradients without mutating W1 or b1."""
        x_train = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        y_one_hot = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )
        forward_pass_results = {
            "Y_one_hot": y_one_hot,
            "A1": np.array(
                [
                    [0.8, 0.2],
                    [0.3, 0.7],
                ],
            ),
        }
        parameters = {
            "W1": np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
            ),
            "b1": np.array([[0.5, -0.5]]),
        }
        original_W1 = parameters["W1"].copy()
        original_b1 = parameters["b1"].copy()

        result = backward_pass.run_backward_pass(
            x_train=x_train,
            forward_pass_results=forward_pass_results,
            parameters=parameters,
            neurons_profile=[2],
            lambda_coefficient=0.0,
        )

        expected_dW1 = np.array(
            [
                [0.35, -0.35],
                [0.4, -0.4],
            ],
        )
        expected_db1 = np.array([[0.05, -0.05]])

        np.testing.assert_allclose(result["gradients"]["dW1"], expected_dW1)
        np.testing.assert_allclose(result["gradients"]["db1"], expected_db1)

        np.testing.assert_array_equal(parameters["W1"], original_W1)
        np.testing.assert_array_equal(parameters["b1"], original_b1)
        self.assertNotIn("parameters", result)

    def test_run_backward_pass_applies_l2_loss_and_weight_gradients(self) -> None:
        """Add L2 loss and L2 weight-gradient terms when lambda is positive."""
        x_train = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        y_one_hot = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )
        forward_pass_results = {
            "Y_one_hot": y_one_hot,
            "A1": np.array(
                [
                    [0.8, 0.2],
                    [0.3, 0.7],
                ],
            ),
        }
        parameters = {
            "W1": np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
            ),
            "b1": np.array([[0.5, -0.5]]),
        }
        original_W1 = parameters["W1"].copy()
        original_b1 = parameters["b1"].copy()

        result = backward_pass.run_backward_pass(
            x_train=x_train,
            forward_pass_results=forward_pass_results,
            parameters=parameters,
            neurons_profile=[2],
            lambda_coefficient=0.2,
        )

        expected_regularized_dW1 = np.array(
            [
                [0.45, -0.15],
                [0.7, 0.0],
            ],
        )

        np.testing.assert_allclose(
            result["gradients"]["dW1"],
            expected_regularized_dW1,
            atol=1e-12,
        )
        np.testing.assert_array_equal(parameters["W1"], original_W1)
        np.testing.assert_array_equal(parameters["b1"], original_b1)

        self.assertAlmostEqual(result["loss"], 1.7899092476264712)
        self.assertNotIn("parameters", result)

    def test_run_backward_pass_applies_l2_using_explicit_regularization_sample_count(
        self,
    ) -> None:
        """Use explicit L2 denominator instead of current batch size."""
        x_batch = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        y_one_hot = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )
        forward_pass_results = {
            "Y_one_hot": y_one_hot,
            "A1": np.array(
                [
                    [0.8, 0.2],
                    [0.3, 0.7],
                ],
            ),
        }
        parameters = {
            "W1": np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
            ),
            "b1": np.array([[0.5, -0.5]]),
        }
        original_W1 = parameters["W1"].copy()
        original_b1 = parameters["b1"].copy()

        result = backward_pass.run_backward_pass(
            x_train=x_batch,
            forward_pass_results=forward_pass_results,
            parameters=parameters,
            neurons_profile=[2],
            lambda_coefficient=0.2,
            regularization_sample_count=8,
        )

        expected_regularized_dW1 = np.array(
            [
                [0.375, -0.3],
                [0.475, -0.3],
            ],
        )

        np.testing.assert_allclose(
            result["gradients"]["dW1"],
            expected_regularized_dW1,
            atol=1e-12,
        )
        np.testing.assert_array_equal(parameters["W1"], original_W1)
        np.testing.assert_array_equal(parameters["b1"], original_b1)

        self.assertAlmostEqual(result["loss"], 0.6649092476264711)
        self.assertNotIn("parameters", result)


if __name__ == "__main__":
    unittest.main()
