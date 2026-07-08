"""Tests for training orchestration utilities."""

import unittest
from unittest.mock import patch

import numpy as np

from training import training


class TestRunInitialTrainingStep(unittest.TestCase):
    """Tests for run_initial_training_step."""

    def test_run_initial_training_step_coordinates_one_full_training_pass(self) -> None:
        """Initialize parameters, run forward pass, and run backward pass."""
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

        Z = np.array(
            [
                [0.7, 1.0],
                [1.5, 2.2],
            ],
        )
        A = np.array(
            [
                [0.4, 0.6],
                [0.7, 0.3],
            ],
        )
        predictions = np.array([1, 0])
        y_one_hot = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
        )

        dZ = np.array(
            [
                [0.4, -0.4],
                [-0.3, 0.3],
            ],
        )
        dW = np.array(
            [
                [-0.25, 0.25],
                [-0.35, 0.35],
            ],
        )
        db = np.array([[0.05, -0.05]])
        updated_W1 = np.array(
            [
                [0.1025, 0.1975],
                [0.3035, 0.3965],
            ],
        )
        updated_b1 = np.array([[-0.0005, 0.0005]])

        with (
            patch.object(
                training,
                "initialize_weights_and_bias",
                return_value={"W1": W1, "b1": b1},
            ) as mock_initialize_weights_and_bias,
            patch.object(
                training,
                "run_forward_pass",
                return_value={
                    "Z": Z,
                    "A": A,
                    "predictions": predictions,
                    "Y_one_hot": y_one_hot,
                },
            ) as mock_run_forward_pass,
            patch.object(
                training,
                "run_backward_pass",
                return_value={
                    "loss": 0.5,
                    "dZ": dZ,
                    "dW": dW,
                    "db": db,
                    "updated_W1": updated_W1,
                    "updated_b1": updated_b1,
                },
            ) as mock_run_backward_pass,
        ):
            result = training.run_initial_training_step(
                x_train=x_train,
                y_train=y_train,
                output_neurons=2,
                learning_rate=0.1,
            )

        self.assertIs(result["W1"], W1)
        self.assertIs(result["b1"], b1)
        self.assertIs(result["Z"], Z)
        self.assertIs(result["A"], A)
        self.assertIs(result["predictions"], predictions)
        self.assertIs(result["Y_one_hot"], y_one_hot)
        self.assertEqual(result["loss"], 0.5)
        self.assertIs(result["dZ"], dZ)
        self.assertIs(result["dW"], dW)
        self.assertIs(result["db"], db)
        self.assertIs(result["updated_W1"], updated_W1)
        self.assertIs(result["updated_b1"], updated_b1)

        mock_initialize_weights_and_bias.assert_called_once_with(
            x_train=x_train,
            h=2,
        )
        mock_run_forward_pass.assert_called_once_with(
            x_train=x_train,
            y_train=y_train,
            W1=W1,
            b1=b1,
        )
        mock_run_backward_pass.assert_called_once_with(
            x_train=x_train,
            y_one_hot=y_one_hot,
            activation=A,
            W1=W1,
            b1=b1,
            learning_rate=0.1,
        )


class TestRunTrainingIterations(unittest.TestCase):
    """Tests for run_training_iterations."""

    def test_run_training_iterations_runs_multiple_training_passes(self) -> None:
        """Run multiple training forward, evaluation, backward, and update steps."""
        x_train = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        y_train = np.array([1, 0])

        initial_W1 = np.array(
            [
                [0.1, 0.2],
                [0.3, 0.4],
            ],
        )
        initial_b1 = np.array([[0.0, 0.0]])

        updated_W1_iteration_1 = np.array(
            [
                [0.11, 0.19],
                [0.31, 0.39],
            ],
        )
        updated_b1_iteration_1 = np.array([[0.01, -0.01]])

        updated_W1_iteration_2 = np.array(
            [
                [0.12, 0.18],
                [0.32, 0.38],
            ],
        )
        updated_b1_iteration_2 = np.array([[0.02, -0.02]])

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
                return_value={"W1": initial_W1, "b1": initial_b1},
            ) as mock_initialize_weights_and_bias,
            patch.object(
                training,
                "run_forward_pass",
                side_effect=[
                    {
                        "Z": np.zeros((2, 2)),
                        "A": activation_iteration_1,
                        "predictions": predictions_iteration_1,
                        "Y_one_hot": y_one_hot,
                    },
                    {
                        "Z": np.zeros((2, 2)),
                        "A": activation_iteration_2,
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
                        "dZ": np.zeros((2, 2)),
                        "dW": np.zeros((2, 2)),
                        "db": np.zeros((1, 2)),
                        "updated_W1": updated_W1_iteration_1,
                        "updated_b1": updated_b1_iteration_1,
                    },
                    {
                        "loss": 0.8,
                        "dZ": np.zeros((2, 2)),
                        "dW": np.zeros((2, 2)),
                        "db": np.zeros((1, 2)),
                        "updated_W1": updated_W1_iteration_2,
                        "updated_b1": updated_b1_iteration_2,
                    },
                ],
            ) as mock_run_backward_pass,
        ):
            result = training.run_training_iterations(
                x_train=x_train,
                y_train=y_train,
                output_neurons=2,
                learning_rate=0.1,
                num_iterations=2,
            )

        self.assertIs(result["final_W1"], updated_W1_iteration_2)
        self.assertIs(result["final_b1"], updated_b1_iteration_2)
        self.assertIs(result["train_predictions"], predictions_iteration_2)
        self.assertEqual(result["validation_predictions"].size, 0)
        self.assertEqual(result["train_loss"], [1.2, 0.8])
        self.assertEqual(result["train_accuracy"], [0.5, 1.0])
        self.assertEqual(result["validation_loss"], [])
        self.assertEqual(result["validation_accuracy"], [])

        mock_initialize_weights_and_bias.assert_called_once_with(
            x_train=x_train,
            h=2,
        )
        self.assertEqual(mock_run_forward_pass.call_count, 2)
        self.assertEqual(mock_run_evaluation.call_count, 2)
        self.assertEqual(mock_run_backward_pass.call_count, 2)

        first_forward_call = mock_run_forward_pass.call_args_list[0].kwargs
        self.assertIs(first_forward_call["W1"], initial_W1)
        self.assertIs(first_forward_call["b1"], initial_b1)

        second_forward_call = mock_run_forward_pass.call_args_list[1].kwargs
        self.assertIs(second_forward_call["W1"], updated_W1_iteration_1)
        self.assertIs(second_forward_call["b1"], updated_b1_iteration_1)

    def test_run_training_iterations_tracks_validation_metrics(self) -> None:
        """Run validation loss and accuracy during each training iteration."""
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

        initial_W1 = np.array(
            [
                [0.1, 0.2],
                [0.3, 0.4],
            ],
        )
        initial_b1 = np.array([[0.0, 0.0]])

        updated_W1_iteration_1 = np.array(
            [
                [0.11, 0.19],
                [0.31, 0.39],
            ],
        )
        updated_b1_iteration_1 = np.array([[0.01, -0.01]])

        updated_W1_iteration_2 = np.array(
            [
                [0.12, 0.18],
                [0.32, 0.38],
            ],
        )
        updated_b1_iteration_2 = np.array([[0.02, -0.02]])

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
                return_value={"W1": initial_W1, "b1": initial_b1},
            ),
            patch.object(
                training,
                "run_forward_pass",
                side_effect=[
                    {
                        "Z": np.zeros((2, 2)),
                        "A": train_activation_iteration_1,
                        "predictions": train_predictions_iteration_1,
                        "Y_one_hot": train_y_one_hot,
                    },
                    {
                        "Z": np.zeros((2, 2)),
                        "A": validation_activation_iteration_1,
                        "predictions": validation_predictions_iteration_1,
                        "Y_one_hot": validation_y_one_hot_iteration_1,
                    },
                    {
                        "Z": np.zeros((2, 2)),
                        "A": train_activation_iteration_2,
                        "predictions": train_predictions_iteration_2,
                        "Y_one_hot": train_y_one_hot,
                    },
                    {
                        "Z": np.zeros((2, 2)),
                        "A": validation_activation_iteration_2,
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
                side_effect=[
                    {"loss": 1.4},
                    {"loss": 0.9},
                ],
            ) as mock_categorical_cross_entropy,
            patch.object(
                training,
                "run_backward_pass",
                side_effect=[
                    {
                        "loss": 1.2,
                        "dZ": np.zeros((2, 2)),
                        "dW": np.zeros((2, 2)),
                        "db": np.zeros((1, 2)),
                        "updated_W1": updated_W1_iteration_1,
                        "updated_b1": updated_b1_iteration_1,
                    },
                    {
                        "loss": 0.8,
                        "dZ": np.zeros((2, 2)),
                        "dW": np.zeros((2, 2)),
                        "db": np.zeros((1, 2)),
                        "updated_W1": updated_W1_iteration_2,
                        "updated_b1": updated_b1_iteration_2,
                    },
                ],
            ),
        ):
            result = training.run_training_iterations(
                x_train=x_train,
                y_train=y_train,
                x_validation=x_validation,
                y_validation=y_validation,
                output_neurons=2,
                learning_rate=0.1,
                num_iterations=2,
            )

        self.assertIs(result["final_W1"], updated_W1_iteration_2)
        self.assertIs(result["final_b1"], updated_b1_iteration_2)
        self.assertIs(result["train_predictions"], train_predictions_iteration_2)
        self.assertIs(
            result["validation_predictions"],
            validation_predictions_iteration_2,
        )
        self.assertEqual(result["train_loss"], [1.2, 0.8])
        self.assertEqual(result["train_accuracy"], [0.5, 1.0])
        self.assertEqual(result["validation_loss"], [1.4, 0.9])
        self.assertEqual(result["validation_accuracy"], [0.25, 0.75])

        self.assertEqual(mock_run_forward_pass.call_count, 4)
        self.assertEqual(mock_run_evaluation.call_count, 4)
        self.assertEqual(mock_categorical_cross_entropy.call_count, 2)

        first_train_forward_call = mock_run_forward_pass.call_args_list[0].kwargs
        first_validation_forward_call = mock_run_forward_pass.call_args_list[1].kwargs
        second_train_forward_call = mock_run_forward_pass.call_args_list[2].kwargs
        second_validation_forward_call = mock_run_forward_pass.call_args_list[3].kwargs

        self.assertIs(first_train_forward_call["W1"], initial_W1)
        self.assertIs(first_train_forward_call["b1"], initial_b1)
        self.assertIs(first_validation_forward_call["W1"], initial_W1)
        self.assertIs(first_validation_forward_call["b1"], initial_b1)

        self.assertIs(second_train_forward_call["W1"], updated_W1_iteration_1)
        self.assertIs(second_train_forward_call["b1"], updated_b1_iteration_1)
        self.assertIs(second_validation_forward_call["W1"], updated_W1_iteration_1)
        self.assertIs(second_validation_forward_call["b1"], updated_b1_iteration_1)

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

    def test_run_training_iterations_raises_error_for_invalid_iterations(self) -> None:
        """Raise ValueError when num_iterations is less than one."""
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
                num_iterations=0,
            )

    def test_run_training_iterations_raises_error_for_partial_validation_data(
        self,
    ) -> None:
        """Raise ValueError when only one validation array is provided."""
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
            )


if __name__ == "__main__":
    unittest.main()
