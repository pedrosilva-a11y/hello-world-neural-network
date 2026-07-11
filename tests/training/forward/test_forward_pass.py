"""Tests for the Digit Recognizer forward pass."""

import unittest
from unittest.mock import patch

import numpy as np

from training.forward import forward_pass


class TestRunForwardPass(unittest.TestCase):
    """Tests for run_forward_pass."""

    def test_run_forward_pass_coordinates_single_layer_softmax_forward_pass(
        self,
    ) -> None:
        """Run pre-activation, softmax, prediction, and one-hot encoding."""
        x_train = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        y_train = np.array([1, 0])
        parameters = {
            "W1": np.array(
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                ],
            ),
            "b1": np.array([[0.01, 0.02]]),
        }
        neurons_profile = [2]

        z1 = np.array(
            [
                [0.7, 1.0],
                [1.5, 2.2],
            ],
        )
        a1 = np.array(
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

        with (
            patch.object(
                forward_pass,
                "compute_pre_activation",
                return_value={"Z1": z1},
            ) as mock_compute_pre_activation,
            patch.object(
                forward_pass,
                "relu",
            ) as mock_relu,
            patch.object(
                forward_pass,
                "softmax",
                return_value=a1,
            ) as mock_softmax,
            patch.object(
                forward_pass,
                "get_predictions",
                return_value=predictions,
            ) as mock_get_predictions,
            patch.object(
                forward_pass,
                "label_one_hot_representation",
                return_value=y_one_hot,
            ) as mock_label_one_hot_representation,
        ):
            result = forward_pass.run_forward_pass(
                x_train=x_train,
                y_train=y_train,
                parameters=parameters,
                neurons_profile=neurons_profile,
            )

        self.assertIs(result["Z1"], z1)
        self.assertIs(result["A1"], a1)
        self.assertIs(result["predictions"], predictions)
        self.assertIs(result["Y_one_hot"], y_one_hot)

        mock_compute_pre_activation.assert_called_once_with(
            a=x_train,
            w=parameters["W1"],
            b=parameters["b1"],
            layer_number=1,
        )
        mock_relu.assert_not_called()
        mock_softmax.assert_called_once_with(logits=z1)
        mock_get_predictions.assert_called_once_with(activation=a1)
        mock_label_one_hot_representation.assert_called_once_with(labels=y_train)

    def test_run_forward_pass_coordinates_one_hidden_relu_layer_forward_pass(
        self,
    ) -> None:
        """Run ReLU for hidden layer and softmax for output layer."""
        x_train = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        y_train = np.array([1, 0])
        parameters = {
            "W1": np.array(
                [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                ],
            ),
            "b1": np.array([[0.01, 0.02, 0.03]]),
            "W2": np.array(
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                    [0.5, 0.6],
                ],
            ),
            "b2": np.array([[0.01, 0.02]]),
        }
        neurons_profile = [3, 2]

        z1 = np.array(
            [
                [0.8, 1.0, 1.2],
                [1.8, 2.0, 2.2],
            ],
        )
        a1 = np.array(
            [
                [0.8, 1.0, 1.2],
                [1.8, 2.0, 2.2],
            ],
        )
        z2 = np.array(
            [
                [1.2, 1.8],
                [2.4, 3.0],
            ],
        )
        a2 = np.array(
            [
                [0.35, 0.65],
                [0.45, 0.55],
            ],
        )
        predictions = np.array([1, 1])
        y_one_hot = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
        )

        with (
            patch.object(
                forward_pass,
                "compute_pre_activation",
                side_effect=[
                    {"Z1": z1},
                    {"Z2": z2},
                ],
            ) as mock_compute_pre_activation,
            patch.object(
                forward_pass,
                "relu",
                return_value=a1,
            ) as mock_relu,
            patch.object(
                forward_pass,
                "softmax",
                return_value=a2,
            ) as mock_softmax,
            patch.object(
                forward_pass,
                "get_predictions",
                return_value=predictions,
            ) as mock_get_predictions,
            patch.object(
                forward_pass,
                "label_one_hot_representation",
                return_value=y_one_hot,
            ) as mock_label_one_hot_representation,
        ):
            result = forward_pass.run_forward_pass(
                x_train=x_train,
                y_train=y_train,
                parameters=parameters,
                neurons_profile=neurons_profile,
            )

        self.assertIs(result["Z1"], z1)
        self.assertIs(result["A1"], a1)
        self.assertIs(result["Z2"], z2)
        self.assertIs(result["A2"], a2)
        self.assertIs(result["predictions"], predictions)
        self.assertIs(result["Y_one_hot"], y_one_hot)

        self.assertEqual(mock_compute_pre_activation.call_count, 2)

        first_call = mock_compute_pre_activation.call_args_list[0].kwargs
        second_call = mock_compute_pre_activation.call_args_list[1].kwargs

        self.assertIs(first_call["a"], x_train)
        self.assertIs(first_call["w"], parameters["W1"])
        self.assertIs(first_call["b"], parameters["b1"])
        self.assertEqual(first_call["layer_number"], 1)

        self.assertIs(second_call["a"], a1)
        self.assertIs(second_call["w"], parameters["W2"])
        self.assertIs(second_call["b"], parameters["b2"])
        self.assertEqual(second_call["layer_number"], 2)

        mock_relu.assert_called_once_with(pre_activation=z1)
        mock_softmax.assert_called_once_with(logits=z2)
        mock_get_predictions.assert_called_once_with(activation=a2)
        mock_label_one_hot_representation.assert_called_once_with(labels=y_train)

    def test_run_forward_pass_returns_expected_shapes_for_single_layer_softmax(
        self,
    ) -> None:
        """Return forward pass outputs with expected shapes for softmax model."""
        x_train = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
        )
        y_train = np.array([1, 0])
        parameters = {
            "W1": np.zeros((3, 10)),
            "b1": np.zeros((1, 10)),
        }

        result = forward_pass.run_forward_pass(
            x_train=x_train,
            y_train=y_train,
            parameters=parameters,
            neurons_profile=[10],
        )

        self.assertEqual(result["Z1"].shape, (2, 10))
        self.assertEqual(result["A1"].shape, (2, 10))
        self.assertEqual(result["predictions"].shape, (2,))
        self.assertEqual(result["Y_one_hot"].shape, (2, 10))

    def test_run_forward_pass_returns_expected_shapes_for_one_hidden_layer(
        self,
    ) -> None:
        """Return forward pass outputs with expected shapes for hidden-layer model."""
        x_train = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
        )
        y_train = np.array([1, 0])
        parameters = {
            "W1": np.zeros((3, 4)),
            "b1": np.zeros((1, 4)),
            "W2": np.zeros((4, 10)),
            "b2": np.zeros((1, 10)),
        }

        result = forward_pass.run_forward_pass(
            x_train=x_train,
            y_train=y_train,
            parameters=parameters,
            neurons_profile=[4, 10],
        )

        self.assertEqual(result["Z1"].shape, (2, 4))
        self.assertEqual(result["A1"].shape, (2, 4))
        self.assertEqual(result["Z2"].shape, (2, 10))
        self.assertEqual(result["A2"].shape, (2, 10))
        self.assertEqual(result["predictions"].shape, (2,))
        self.assertEqual(result["Y_one_hot"].shape, (2, 10))


if __name__ == "__main__":
    unittest.main()
