"""Tests for training orchestration utilities."""

import unittest
from unittest.mock import patch

import numpy as np

from training import training


def _training_config(
    num_iterations: int = 1,
    learning_rate: float = 0.1,
    optimizer: str = "batch_gradient_descent",
    regularization_enabled: bool = False,
    regularization_type: str = "none",
    lambda_coefficient: float = 0.0,
) -> dict:
    """Create a training config with regularization fields."""
    return {
        "optimizer": optimizer,
        "num_iterations": num_iterations,
        "learning_rate": learning_rate,
        "regularization": {
            "enabled": regularization_enabled,
            "type": regularization_type,
            "lambda": lambda_coefficient,
        },
    }


class TestRunInitialTrainingStep(unittest.TestCase):
    """Tests for run_initial_training_step."""

    def test_run_initial_training_step_coordinates_one_full_training_pass(self) -> None:
        """Initialize parameters, run forward pass, and run backward pass."""
        model_config = {
            "name": "single_layer_softmax_classifier",
            "input_size": 2,
            "neurons_profile": [2],
        }
        training_config = _training_config(
            regularization_enabled=True,
            regularization_type="l2",
            lambda_coefficient=0.2,
        )

        x_train = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        y_train = np.array([1, 0])

        W1 = np.array(
            [
                [0.1, 0.2],
                [0.3, 0.4],
            ],
        )
        b1 = np.array([[0.0, 0.0]])
        parameters = {
            "W1": W1,
            "b1": b1,
        }

        forward_output = {
            "Z1": np.array(
                [
                    [0.7, 1.0],
                    [1.5, 2.2],
                ],
            ),
            "A1": np.array(
                [
                    [0.4, 0.6],
                    [0.7, 0.3],
                ],
            ),
            "predictions": np.array([1, 0]),
            "Y_one_hot": np.array(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                ],
            ),
        }

        gradients = {
            "dZ1": np.array(
                [
                    [0.4, -0.4],
                    [-0.3, 0.3],
                ],
            ),
            "dW1": np.array(
                [
                    [-0.25, 0.25],
                    [-0.35, 0.35],
                ],
            ),
            "db1": np.array([[0.05, -0.05]]),
        }
        updated_parameters = {
            "W1": np.array(
                [
                    [0.1025, 0.1975],
                    [0.3035, 0.3965],
                ],
            ),
            "b1": np.array([[-0.0005, 0.0005]]),
        }
        backward_output = {
            "loss": 0.5,
            "gradients": gradients,
            "parameters": updated_parameters,
        }

        with (
            patch.object(
                training,
                "initialize_weights_and_bias",
                return_value=parameters,
            ) as mock_initialize_weights_and_bias,
            patch.object(
                training,
                "run_forward_pass",
                return_value=forward_output,
            ) as mock_run_forward_pass,
            patch.object(
                training,
                "run_backward_pass",
                return_value=backward_output,
            ) as mock_run_backward_pass,
        ):
            result = training.run_initial_training_step(
                x_train=x_train,
                y_train=y_train,
                model_config=model_config,
                training_config=training_config,
            )

        np.testing.assert_array_equal(result["initial_parameters"]["W1"], W1)
        np.testing.assert_array_equal(result["initial_parameters"]["b1"], b1)
        self.assertIs(result["forward_output"], forward_output)
        self.assertIs(result["backward_output"], backward_output)
        self.assertIs(result["updated_parameters"], updated_parameters)

        mock_initialize_weights_and_bias.assert_called_once_with(
            x_train=x_train,
            neurons_profile=[2],
        )
        mock_run_forward_pass.assert_called_once_with(
            x_train=x_train,
            y_train=y_train,
            parameters=parameters,
            neurons_profile=[2],
        )
        mock_run_backward_pass.assert_called_once_with(
            x_train=x_train,
            forward_pass_results=forward_output,
            parameters=parameters,
            neurons_profile=[2],
            lambda_coefficient=0.2,
            learning_rate=0.1,
        )


class TestRunTrainingIterations(unittest.TestCase):
    """Tests for run_training_iterations."""

    def test_run_training_iterations_runs_multiple_training_passes(self) -> None:
        """Run multiple training forward, evaluation, backward, and update steps."""
        model_config = {
            "name": "single_layer_softmax_classifier",
            "input_size": 2,
            "neurons_profile": [2],
        }
        training_config = _training_config(
            num_iterations=2,
            regularization_enabled=True,
            regularization_type="l2",
            lambda_coefficient=0.25,
        )

        x_train = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        y_train = np.array([1, 0])

        initial_parameters = {
            "W1": np.array(
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                ],
            ),
            "b1": np.array([[0.0, 0.0]]),
        }
        updated_parameters_iteration_1 = {
            "W1": np.array(
                [
                    [0.11, 0.19],
                    [0.31, 0.39],
                ],
            ),
            "b1": np.array([[0.01, -0.01]]),
        }
        updated_parameters_iteration_2 = {
            "W1": np.array(
                [
                    [0.12, 0.18],
                    [0.32, 0.38],
                ],
            ),
            "b1": np.array([[0.02, -0.02]]),
        }

        y_one_hot = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
        )
        activation_iteration_1 = np.array(
            [
                [0.4, 0.6],
                [0.7, 0.3],
            ],
        )
        activation_iteration_2 = np.array(
            [
                [0.3, 0.7],
                [0.8, 0.2],
            ],
        )
        predictions_iteration_1 = np.array([1, 0])
        predictions_iteration_2 = np.array([1, 0])

        with (
            patch.object(
                training,
                "initialize_weights_and_bias",
                return_value=initial_parameters,
            ) as mock_initialize_weights_and_bias,
            patch.object(
                training,
                "run_forward_pass",
                side_effect=[
                    {
                        "Z1": np.zeros((2, 2)),
                        "A1": activation_iteration_1,
                        "predictions": predictions_iteration_1,
                        "Y_one_hot": y_one_hot,
                    },
                    {
                        "Z1": np.zeros((2, 2)),
                        "A1": activation_iteration_2,
                        "predictions": predictions_iteration_2,
                        "Y_one_hot": y_one_hot,
                    },
                ],
            ) as mock_run_forward_pass,
            patch.object(
                training,
                "run_evaluation",
                side_effect=[
                    {"accuracy": 0.5},
                    {"accuracy": 1.0},
                ],
            ) as mock_run_evaluation,
            patch.object(
                training,
                "run_backward_pass",
                side_effect=[
                    {
                        "loss": 1.2,
                        "gradients": {
                            "dZ1": np.zeros((2, 2)),
                            "dW1": np.zeros((2, 2)),
                            "db1": np.zeros((1, 2)),
                        },
                        "parameters": updated_parameters_iteration_1,
                    },
                    {
                        "loss": 0.8,
                        "gradients": {
                            "dZ1": np.zeros((2, 2)),
                            "dW1": np.zeros((2, 2)),
                            "db1": np.zeros((1, 2)),
                        },
                        "parameters": updated_parameters_iteration_2,
                    },
                ],
            ) as mock_run_backward_pass,
        ):
            result = training.run_training_iterations(
                x_train=x_train,
                y_train=y_train,
                model_config=model_config,
                training_config=training_config,
            )

        self.assertIs(result["final_parameters"], updated_parameters_iteration_2)
        self.assertIs(result["train_predictions"], predictions_iteration_2)
        self.assertEqual(result["validation_predictions"].size, 0)
        self.assertEqual(result["train_loss"], [1.2, 0.8])
        self.assertEqual(result["train_accuracy"], [0.5, 1.0])
        self.assertEqual(result["validation_loss"], [])
        self.assertEqual(result["validation_accuracy"], [])

        mock_initialize_weights_and_bias.assert_called_once_with(
            x_train=x_train,
            neurons_profile=[2],
        )
        self.assertEqual(mock_run_forward_pass.call_count, 2)
        self.assertEqual(mock_run_evaluation.call_count, 2)
        self.assertEqual(mock_run_backward_pass.call_count, 2)

        first_forward_call = mock_run_forward_pass.call_args_list[0].kwargs
        self.assertIs(first_forward_call["x_train"], x_train)
        self.assertIs(first_forward_call["y_train"], y_train)
        self.assertIs(first_forward_call["parameters"], initial_parameters)
        self.assertEqual(first_forward_call["neurons_profile"], [2])

        second_forward_call = mock_run_forward_pass.call_args_list[1].kwargs
        self.assertIs(second_forward_call["x_train"], x_train)
        self.assertIs(second_forward_call["y_train"], y_train)
        self.assertIs(
            second_forward_call["parameters"],
            updated_parameters_iteration_1,
        )
        self.assertEqual(second_forward_call["neurons_profile"], [2])

        first_backward_call = mock_run_backward_pass.call_args_list[0].kwargs
        second_backward_call = mock_run_backward_pass.call_args_list[1].kwargs

        self.assertEqual(first_backward_call["lambda_coefficient"], 0.25)
        self.assertEqual(second_backward_call["lambda_coefficient"], 0.25)

    def test_run_training_iterations_tracks_validation_metrics(self) -> None:
        """Run validation loss and accuracy during each training iteration."""
        model_config = {
            "name": "single_layer_softmax_classifier",
            "input_size": 2,
            "neurons_profile": [2],
        }
        training_config = _training_config(num_iterations=2)

        x_train = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        y_train = np.array([1, 0])
        x_validation = np.array(
            [
                [5.0, 6.0],
                [7.0, 8.0],
            ],
        )
        y_validation = np.array([0, 1])

        initial_parameters = {
            "W1": np.array(
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                ],
            ),
            "b1": np.array([[0.0, 0.0]]),
        }
        updated_parameters_iteration_1 = {
            "W1": np.array(
                [
                    [0.11, 0.19],
                    [0.31, 0.39],
                ],
            ),
            "b1": np.array([[0.01, -0.01]]),
        }
        updated_parameters_iteration_2 = {
            "W1": np.array(
                [
                    [0.12, 0.18],
                    [0.32, 0.38],
                ],
            ),
            "b1": np.array([[0.02, -0.02]]),
        }

        train_y_one_hot = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
        )
        validation_y_one_hot_iteration_1 = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )
        validation_y_one_hot_iteration_2 = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )

        train_activation_iteration_1 = np.array(
            [
                [0.4, 0.6],
                [0.7, 0.3],
            ],
        )
        validation_activation_iteration_1 = np.array(
            [
                [0.7, 0.3],
                [0.6, 0.4],
            ],
        )
        train_activation_iteration_2 = np.array(
            [
                [0.3, 0.7],
                [0.8, 0.2],
            ],
        )
        validation_activation_iteration_2 = np.array(
            [
                [0.8, 0.2],
                [0.3, 0.7],
            ],
        )

        train_predictions_iteration_1 = np.array([1, 0])
        validation_predictions_iteration_1 = np.array([0, 0])
        train_predictions_iteration_2 = np.array([1, 0])
        validation_predictions_iteration_2 = np.array([0, 1])

        with (
            patch.object(
                training,
                "initialize_weights_and_bias",
                return_value=initial_parameters,
            ) as mock_initialize_weights_and_bias,
            patch.object(
                training,
                "run_forward_pass",
                side_effect=[
                    {
                        "Z1": np.zeros((2, 2)),
                        "A1": train_activation_iteration_1,
                        "predictions": train_predictions_iteration_1,
                        "Y_one_hot": train_y_one_hot,
                    },
                    {
                        "Z1": np.zeros((2, 2)),
                        "A1": validation_activation_iteration_1,
                        "predictions": validation_predictions_iteration_1,
                        "Y_one_hot": validation_y_one_hot_iteration_1,
                    },
                    {
                        "Z1": np.zeros((2, 2)),
                        "A1": train_activation_iteration_2,
                        "predictions": train_predictions_iteration_2,
                        "Y_one_hot": train_y_one_hot,
                    },
                    {
                        "Z1": np.zeros((2, 2)),
                        "A1": validation_activation_iteration_2,
                        "predictions": validation_predictions_iteration_2,
                        "Y_one_hot": validation_y_one_hot_iteration_2,
                    },
                ],
            ) as mock_run_forward_pass,
            patch.object(
                training,
                "run_evaluation",
                side_effect=[
                    {"accuracy": 0.5},
                    {"accuracy": 0.25},
                    {"accuracy": 1.0},
                    {"accuracy": 0.75},
                ],
            ) as mock_run_evaluation,
            patch.object(
                training,
                "categorical_cross_entropy",
                side_effect=[1.4, 0.9],
            ) as mock_categorical_cross_entropy,
            patch.object(
                training,
                "run_backward_pass",
                side_effect=[
                    {
                        "loss": 1.2,
                        "gradients": {
                            "dZ1": np.zeros((2, 2)),
                            "dW1": np.zeros((2, 2)),
                            "db1": np.zeros((1, 2)),
                        },
                        "parameters": updated_parameters_iteration_1,
                    },
                    {
                        "loss": 0.8,
                        "gradients": {
                            "dZ1": np.zeros((2, 2)),
                            "dW1": np.zeros((2, 2)),
                            "db1": np.zeros((1, 2)),
                        },
                        "parameters": updated_parameters_iteration_2,
                    },
                ],
            ) as mock_run_backward_pass,
        ):
            result = training.run_training_iterations(
                x_train=x_train,
                y_train=y_train,
                x_validation=x_validation,
                y_validation=y_validation,
                model_config=model_config,
                training_config=training_config,
            )

        self.assertIs(result["final_parameters"], updated_parameters_iteration_2)
        self.assertIs(result["train_predictions"], train_predictions_iteration_2)
        self.assertIs(
            result["validation_predictions"],
            validation_predictions_iteration_2,
        )
        self.assertEqual(result["train_loss"], [1.2, 0.8])
        self.assertEqual(result["train_accuracy"], [0.5, 1.0])
        self.assertEqual(result["validation_loss"], [1.4, 0.9])
        self.assertEqual(result["validation_accuracy"], [0.25, 0.75])

        mock_initialize_weights_and_bias.assert_called_once_with(
            x_train=x_train,
            neurons_profile=[2],
        )
        self.assertEqual(mock_run_forward_pass.call_count, 4)
        self.assertEqual(mock_run_evaluation.call_count, 4)
        self.assertEqual(mock_categorical_cross_entropy.call_count, 2)
        self.assertEqual(mock_run_backward_pass.call_count, 2)

        first_train_forward_call = mock_run_forward_pass.call_args_list[0].kwargs
        first_validation_forward_call = mock_run_forward_pass.call_args_list[1].kwargs
        second_train_forward_call = mock_run_forward_pass.call_args_list[2].kwargs
        second_validation_forward_call = mock_run_forward_pass.call_args_list[3].kwargs

        self.assertIs(first_train_forward_call["x_train"], x_train)
        self.assertIs(first_train_forward_call["y_train"], y_train)
        self.assertIs(first_train_forward_call["parameters"], initial_parameters)

        self.assertIs(first_validation_forward_call["x_train"], x_validation)
        self.assertIs(first_validation_forward_call["y_train"], y_validation)
        self.assertIs(first_validation_forward_call["parameters"], initial_parameters)

        self.assertIs(second_train_forward_call["x_train"], x_train)
        self.assertIs(second_train_forward_call["y_train"], y_train)
        self.assertIs(
            second_train_forward_call["parameters"],
            updated_parameters_iteration_1,
        )

        self.assertIs(second_validation_forward_call["x_train"], x_validation)
        self.assertIs(second_validation_forward_call["y_train"], y_validation)
        self.assertIs(
            second_validation_forward_call["parameters"],
            updated_parameters_iteration_1,
        )

        first_validation_loss_call = mock_categorical_cross_entropy.call_args_list[
            0
        ].kwargs
        second_validation_loss_call = mock_categorical_cross_entropy.call_args_list[
            1
        ].kwargs

        self.assertIs(
            first_validation_loss_call["y_one_hot"],
            validation_y_one_hot_iteration_1,
        )
        self.assertIs(
            first_validation_loss_call["y_pred"],
            validation_activation_iteration_1,
        )
        self.assertIs(
            second_validation_loss_call["y_one_hot"],
            validation_y_one_hot_iteration_2,
        )
        self.assertIs(
            second_validation_loss_call["y_pred"],
            validation_activation_iteration_2,
        )

        first_backward_call = mock_run_backward_pass.call_args_list[0].kwargs
        second_backward_call = mock_run_backward_pass.call_args_list[1].kwargs

        self.assertEqual(first_backward_call["lambda_coefficient"], 0.0)
        self.assertEqual(second_backward_call["lambda_coefficient"], 0.0)

    def test_run_training_iterations_raises_error_for_invalid_iterations(self) -> None:
        """Raise ValueError when num_iterations is less than one."""
        model_config = {
            "name": "single_layer_softmax_classifier",
            "input_size": 2,
            "neurons_profile": [2],
        }
        training_config = _training_config(num_iterations=0)

        x_train = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        y_train = np.array([1, 0])

        with self.assertRaisesRegex(ValueError, "num_iterations must be at least 1."):
            training.run_training_iterations(
                x_train=x_train,
                y_train=y_train,
                model_config=model_config,
                training_config=training_config,
            )

    def test_run_training_iterations_raises_error_for_partial_validation_data(
        self,
    ) -> None:
        """Raise ValueError when only one validation array is provided."""
        model_config = {
            "name": "single_layer_softmax_classifier",
            "input_size": 2,
            "neurons_profile": [2],
        }
        training_config = _training_config(num_iterations=2)

        x_train = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        y_train = np.array([1, 0])
        x_validation = np.array(
            [
                [5.0, 6.0],
                [7.0, 8.0],
            ],
        )

        with self.assertRaisesRegex(
            ValueError,
            "x_validation and y_validation must be provided together.",
        ):
            training.run_training_iterations(
                x_train=x_train,
                y_train=y_train,
                x_validation=x_validation,
                model_config=model_config,
                training_config=training_config,
            )

    def test_run_training_iterations_raises_error_for_unsupported_model_name(
        self,
    ) -> None:
        """Raise ValueError when the configured model is unsupported."""
        model_config = {
            "name": "unsupported_classifier",
            "input_size": 2,
            "neurons_profile": [2],
        }
        training_config = _training_config(num_iterations=2)

        x_train = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        y_train = np.array([1, 0])

        with self.assertRaisesRegex(
            ValueError,
            "Unsupported model name: unsupported_classifier",
        ):
            training.run_training_iterations(
                x_train=x_train,
                y_train=y_train,
                model_config=model_config,
                training_config=training_config,
            )

    def test_run_training_iterations_accepts_hidden_layer_relu_model(self) -> None:
        """Accept the one-hidden-layer ReLU model configuration."""
        model_config = {
            "name": "one_hidden_layer_relu_classifier",
            "input_size": 2,
            "neurons_profile": [3, 2],
        }
        training_config = _training_config()

        x_train = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        y_train = np.array([1, 0])
        parameters = {
            "W1": np.zeros((2, 3)),
            "b1": np.zeros((1, 3)),
            "W2": np.zeros((3, 2)),
            "b2": np.zeros((1, 2)),
        }

        with (
            patch.object(
                training,
                "initialize_weights_and_bias",
                return_value=parameters,
            ) as mock_initialize_weights_and_bias,
            patch.object(
                training,
                "run_forward_pass",
                return_value={
                    "Z1": np.zeros((2, 3)),
                    "A1": np.zeros((2, 3)),
                    "Z2": np.zeros((2, 2)),
                    "A2": np.array(
                        [
                            [0.4, 0.6],
                            [0.7, 0.3],
                        ],
                    ),
                    "predictions": np.array([1, 0]),
                    "Y_one_hot": np.array(
                        [
                            [0.0, 1.0],
                            [1.0, 0.0],
                        ],
                    ),
                },
            ) as mock_run_forward_pass,
            patch.object(
                training,
                "run_evaluation",
                return_value={"accuracy": 1.0},
            ),
            patch.object(
                training,
                "run_backward_pass",
                return_value={
                    "loss": 0.5,
                    "gradients": {},
                    "parameters": parameters,
                },
            ) as mock_run_backward_pass,
        ):
            result = training.run_training_iterations(
                x_train=x_train,
                y_train=y_train,
                model_config=model_config,
                training_config=training_config,
            )

        self.assertIs(result["final_parameters"], parameters)
        self.assertEqual(result["train_loss"], [0.5])
        self.assertEqual(result["train_accuracy"], [1.0])

        mock_initialize_weights_and_bias.assert_called_once_with(
            x_train=x_train,
            neurons_profile=[3, 2],
        )
        mock_run_forward_pass.assert_called_once_with(
            x_train=x_train,
            y_train=y_train,
            parameters=parameters,
            neurons_profile=[3, 2],
        )
        mock_run_backward_pass.assert_called_once_with(
            x_train=x_train,
            forward_pass_results=mock_run_forward_pass.return_value,
            parameters=parameters,
            neurons_profile=[3, 2],
            lambda_coefficient=0.0,
            learning_rate=0.1,
        )

    def test_run_training_iterations_accepts_multi_layer_relu_model(self) -> None:
        """Accept the multi-layer ReLU model configuration."""
        model_config = {
            "name": "multi_layer_relu_classifier",
            "input_size": 2,
            "neurons_profile": [3, 3, 2],
        }
        training_config = _training_config()

        x_train = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        y_train = np.array([1, 0])
        parameters = {
            "W1": np.zeros((2, 3)),
            "b1": np.zeros((1, 3)),
            "W2": np.zeros((3, 3)),
            "b2": np.zeros((1, 3)),
            "W3": np.zeros((3, 2)),
            "b3": np.zeros((1, 2)),
        }

        with (
            patch.object(
                training,
                "initialize_weights_and_bias",
                return_value=parameters,
            ) as mock_initialize_weights_and_bias,
            patch.object(
                training,
                "run_forward_pass",
                return_value={
                    "Z1": np.zeros((2, 3)),
                    "A1": np.zeros((2, 3)),
                    "Z2": np.zeros((2, 3)),
                    "A2": np.zeros((2, 3)),
                    "Z3": np.zeros((2, 2)),
                    "A3": np.array(
                        [
                            [0.4, 0.6],
                            [0.7, 0.3],
                        ],
                    ),
                    "predictions": np.array([1, 0]),
                    "Y_one_hot": np.array(
                        [
                            [0.0, 1.0],
                            [1.0, 0.0],
                        ],
                    ),
                },
            ) as mock_run_forward_pass,
            patch.object(
                training,
                "run_evaluation",
                return_value={"accuracy": 1.0},
            ),
            patch.object(
                training,
                "run_backward_pass",
                return_value={
                    "loss": 0.5,
                    "gradients": {},
                    "parameters": parameters,
                },
            ) as mock_run_backward_pass,
        ):
            result = training.run_training_iterations(
                x_train=x_train,
                y_train=y_train,
                model_config=model_config,
                training_config=training_config,
            )

        self.assertIs(result["final_parameters"], parameters)
        self.assertEqual(result["train_loss"], [0.5])
        self.assertEqual(result["train_accuracy"], [1.0])

        mock_initialize_weights_and_bias.assert_called_once_with(
            x_train=x_train,
            neurons_profile=[3, 3, 2],
        )
        mock_run_forward_pass.assert_called_once_with(
            x_train=x_train,
            y_train=y_train,
            parameters=parameters,
            neurons_profile=[3, 3, 2],
        )
        mock_run_backward_pass.assert_called_once_with(
            x_train=x_train,
            forward_pass_results=mock_run_forward_pass.return_value,
            parameters=parameters,
            neurons_profile=[3, 3, 2],
            lambda_coefficient=0.0,
            learning_rate=0.1,
        )

    def test_run_training_iterations_raises_error_for_unsupported_optimizer(
        self,
    ) -> None:
        """Raise ValueError when the configured optimizer is unsupported."""
        model_config = {
            "name": "single_layer_softmax_classifier",
            "input_size": 2,
            "neurons_profile": [2],
        }
        training_config = _training_config(
            optimizer="adam",
            num_iterations=2,
        )

        x_train = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        y_train = np.array([1, 0])

        with self.assertRaisesRegex(ValueError, "Unsupported optimizer: adam"):
            training.run_training_iterations(
                x_train=x_train,
                y_train=y_train,
                model_config=model_config,
                training_config=training_config,
            )


if __name__ == "__main__":
    unittest.main()
