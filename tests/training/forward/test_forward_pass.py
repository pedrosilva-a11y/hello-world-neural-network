"""Tests for the Digit Recognizer forward pass."""

import unittest
from unittest.mock import patch

import numpy as np

from training.forward import forward_pass


class TestRunForwardPass(unittest.TestCase):
    """Tests for run_forward_pass."""

    def test_run_forward_pass_coordinates_forward_computations(self) -> None:
        """Run pre-activation, softmax, prediction, and one-hot encoding."""
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
        b1 = np.array([[0.01, 0.02]])

        z = np.array(
            [
                [0.7, 1.0],
                [1.5, 2.2],
            ],
        )
        activation = np.array(
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
                return_value={"Z": z},
            ) as mock_compute_pre_activation,
            patch.object(
                forward_pass,
                "softmax",
                return_value={"A": activation},
            ) as mock_softmax,
            patch.object(
                forward_pass,
                "get_predictions",
                return_value={"predictions": predictions},
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
                W1=W1,
                b1=b1,
            )

        self.assertIs(result["Z"], z)
        self.assertIs(result["A"], activation)
        self.assertIs(result["predictions"], predictions)
        self.assertIs(result["Y_one_hot"], y_one_hot)

        mock_compute_pre_activation.assert_called_once_with(
            x=x_train,
            w=W1,
            b=b1,
        )
        mock_softmax.assert_called_once_with(logits=z)
        mock_get_predictions.assert_called_once_with(activation=activation)
        mock_label_one_hot_representation.assert_called_once_with(labels=y_train)

    def test_run_forward_pass_returns_expected_shapes(self) -> None:
        """Return forward pass outputs with expected shapes."""
        x_train = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
        )
        y_train = np.array([1, 0])
        W1 = np.zeros((3, 10))
        b1 = np.zeros((1, 10))

        result = forward_pass.run_forward_pass(
            x_train=x_train,
            y_train=y_train,
            W1=W1,
            b1=b1,
        )

        self.assertEqual(result["Z"].shape, (2, 10))
        self.assertEqual(result["A"].shape, (2, 10))
        self.assertEqual(result["predictions"].shape, (2,))
        self.assertEqual(result["Y_one_hot"].shape, (2, 10))


if __name__ == "__main__":
    unittest.main()
