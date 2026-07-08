"""Tests for the Digit Recognizer backward pass."""

import unittest
from unittest.mock import patch

import numpy as np

from training.backpropagation import backward_pass


class TestRunBackwardPass(unittest.TestCase):
    """Tests for run_backward_pass."""

    def test_run_backward_pass_coordinates_backward_computations(self) -> None:
        """Run loss, gradient, and parameter-update computations."""
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
        activation = np.array(
            [
                [0.8, 0.2],
                [0.3, 0.7],
            ],
        )
        W1 = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        b1 = np.array([[0.5, -0.5]])

        dZ = np.array(
            [
                [-0.2, 0.2],
                [0.3, -0.3],
            ],
        )
        dW = np.array(
            [
                [0.35, -0.35],
                [0.4, -0.4],
            ],
        )
        db = np.array([[0.05, -0.05]])
        updated_W1 = np.array(
            [
                [0.965, 2.035],
                [2.96, 4.04],
            ],
        )
        updated_b1 = np.array([[0.495, -0.495]])

        with (
            patch.object(
                backward_pass,
                "categorical_cross_entropy",
                return_value={"loss": 0.289909},
            ) as mock_categorical_cross_entropy,
            patch.object(
                backward_pass,
                "gradient_computations_softmax",
                return_value={"dZ": dZ, "dW": dW, "db": db},
            ) as mock_gradient_computations_softmax,
            patch.object(
                backward_pass,
                "batch_gradient_descent",
                return_value={"W1": updated_W1, "b1": updated_b1},
            ) as mock_batch_gradient_descent,
        ):
            result = backward_pass.run_backward_pass(
                x_train=x_train,
                y_one_hot=y_one_hot,
                activation=activation,
                W1=W1,
                b1=b1,
                learning_rate=0.1,
            )

        self.assertEqual(result["loss"], 0.289909)
        self.assertIs(result["dZ"], dZ)
        self.assertIs(result["dW"], dW)
        self.assertIs(result["db"], db)
        self.assertIs(result["updated_W1"], updated_W1)
        self.assertIs(result["updated_b1"], updated_b1)

        mock_categorical_cross_entropy.assert_called_once_with(
            y_one_hot=y_one_hot,
            y_pred=activation,
        )
        mock_gradient_computations_softmax.assert_called_once_with(
            x=x_train,
            yhot=y_one_hot,
            activation=activation,
        )
        mock_batch_gradient_descent.assert_called_once_with(
            W1=W1,
            b1=b1,
            dW1=dW,
            db1=db,
            learning_rate=0.1,
        )

    def test_run_backward_pass_returns_expected_shapes(self) -> None:
        """Return gradients and updated parameters with expected shapes."""
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
        activation = np.array(
            [
                [0.8, 0.2],
                [0.3, 0.7],
            ],
        )
        W1 = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        b1 = np.array([[0.5, -0.5]])

        result = backward_pass.run_backward_pass(
            x_train=x_train,
            y_one_hot=y_one_hot,
            activation=activation,
            W1=W1,
            b1=b1,
            learning_rate=0.1,
        )

        self.assertIsInstance(result["loss"], float)
        self.assertEqual(result["dZ"].shape, (2, 2))
        self.assertEqual(result["dW"].shape, (2, 2))
        self.assertEqual(result["db"].shape, (1, 2))
        self.assertEqual(result["updated_W1"].shape, (2, 2))
        self.assertEqual(result["updated_b1"].shape, (1, 2))

    def test_run_backward_pass_updates_parameters_correctly(self) -> None:
        """Return updated W1 and b1 after one gradient descent step."""
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
        activation = np.array(
            [
                [0.8, 0.2],
                [0.3, 0.7],
            ],
        )
        W1 = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        b1 = np.array([[0.5, -0.5]])

        result = backward_pass.run_backward_pass(
            x_train=x_train,
            y_one_hot=y_one_hot,
            activation=activation,
            W1=W1,
            b1=b1,
            learning_rate=0.1,
        )

        expected_updated_W1 = np.array(
            [
                [0.965, 2.035],
                [2.96, 4.04],
            ],
        )
        expected_updated_b1 = np.array([[0.495, -0.495]])

        np.testing.assert_allclose(result["updated_W1"], expected_updated_W1)
        np.testing.assert_allclose(result["updated_b1"], expected_updated_b1)


if __name__ == "__main__":
    unittest.main()
