"""Tests for categorical cross-entropy gradient computations."""

import unittest

import numpy as np

from training.backpropagation.gradient import gradients_cross_entropy


class TestGradientComputationsSoftmax(unittest.TestCase):
    """Tests for gradient_computations_softmax."""

    def test_gradient_computations_softmax_returns_expected_layer_keys(self) -> None:
        """Return a dictionary containing layer-specific dZ, dW, and db."""
        x = np.array([[1.0, 2.0]])
        yhot = np.array([[1.0, 0.0]])
        forward_pass_results = {
            "A1": np.array([[0.8, 0.2]]),
        }

        result = gradients_cross_entropy.gradient_computations_softmax(
            x=x,
            yhot=yhot,
            forward_pass_results=forward_pass_results,
            layer=1,
        )

        self.assertEqual(set(result.keys()), {"dZ1", "dW1", "db1"})

    def test_gradient_computations_softmax_computes_dz_for_output_layer(self) -> None:
        """Compute dZ as output activation minus one-hot labels."""
        x = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        yhot = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )
        forward_pass_results = {
            "A1": np.array(
                [
                    [0.8, 0.2],
                    [0.3, 0.7],
                ],
            ),
        }

        result = gradients_cross_entropy.gradient_computations_softmax(
            x=x,
            yhot=yhot,
            forward_pass_results=forward_pass_results,
            layer=1,
        )

        expected_dz = np.array(
            [
                [-0.2, 0.2],
                [0.3, -0.3],
            ],
        )

        np.testing.assert_allclose(result["dZ1"], expected_dz)

    def test_gradient_computations_softmax_computes_dw_for_first_layer(self) -> None:
        """Compute dW using X as the previous activation for layer 1."""
        x = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        yhot = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )
        forward_pass_results = {
            "A1": np.array(
                [
                    [0.8, 0.2],
                    [0.3, 0.7],
                ],
            ),
        }

        result = gradients_cross_entropy.gradient_computations_softmax(
            x=x,
            yhot=yhot,
            forward_pass_results=forward_pass_results,
            layer=1,
        )

        expected_dw = np.array(
            [
                [0.35, -0.35],
                [0.4, -0.4],
            ],
        )

        np.testing.assert_allclose(result["dW1"], expected_dw)

    def test_gradient_computations_softmax_computes_dw_for_later_layer(self) -> None:
        """Compute dW using the previous layer activation for layers after 1."""
        x = np.zeros((2, 2))
        yhot = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )
        forward_pass_results = {
            "A1": np.array(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ],
            ),
            "A2": np.array(
                [
                    [0.8, 0.2],
                    [0.3, 0.7],
                ],
            ),
        }

        result = gradients_cross_entropy.gradient_computations_softmax(
            x=x,
            yhot=yhot,
            forward_pass_results=forward_pass_results,
            layer=2,
        )

        expected_dw = np.array(
            [
                [0.5, -0.5],
                [0.55, -0.55],
                [0.6, -0.6],
            ],
        )

        np.testing.assert_allclose(result["dW2"], expected_dw)

    def test_gradient_computations_softmax_computes_db(self) -> None:
        """Compute db as the average dZ value per output class."""
        x = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        yhot = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )
        forward_pass_results = {
            "A1": np.array(
                [
                    [0.8, 0.2],
                    [0.3, 0.7],
                ],
            ),
        }

        result = gradients_cross_entropy.gradient_computations_softmax(
            x=x,
            yhot=yhot,
            forward_pass_results=forward_pass_results,
            layer=1,
        )

        expected_db = np.array([[0.05, -0.05]])

        np.testing.assert_allclose(result["db1"], expected_db)

    def test_gradient_computations_softmax_returns_expected_shapes(self) -> None:
        """Return gradients with shapes matching Z, W, and b."""
        x = np.zeros((5, 3))
        yhot = np.zeros((5, 4))
        forward_pass_results = {
            "A1": np.zeros((5, 4)),
        }

        result = gradients_cross_entropy.gradient_computations_softmax(
            x=x,
            yhot=yhot,
            forward_pass_results=forward_pass_results,
            layer=1,
        )

        self.assertEqual(result["dZ1"].shape, (5, 4))
        self.assertEqual(result["dW1"].shape, (3, 4))
        self.assertEqual(result["db1"].shape, (1, 4))


class TestReluDerivative(unittest.TestCase):
    """Tests for _relu_derivative."""

    def test_relu_derivative_returns_one_for_positive_values(self) -> None:
        """Return one where pre-activation values are positive."""
        pre_activation = np.array([[1.0, 2.0, 3.0]])

        result = gradients_cross_entropy._relu_derivative(
            pre_activation=pre_activation,
        )

        expected = np.array([[1, 1, 1]])

        np.testing.assert_array_equal(result, expected)

    def test_relu_derivative_returns_zero_for_negative_and_zero_values(self) -> None:
        """Return zero where pre-activation values are negative or zero."""
        pre_activation = np.array([[-2.0, -1.0, 0.0]])

        result = gradients_cross_entropy._relu_derivative(
            pre_activation=pre_activation,
        )

        expected = np.array([[0, 0, 0]])

        np.testing.assert_array_equal(result, expected)

    def test_relu_derivative_applies_element_wise_to_matrix(self) -> None:
        """Apply the ReLU derivative element-wise to a matrix."""
        pre_activation = np.array(
            [
                [-1.0, 0.0, 2.0],
                [3.0, -4.0, 5.0],
            ],
        )

        result = gradients_cross_entropy._relu_derivative(
            pre_activation=pre_activation,
        )

        expected = np.array(
            [
                [0, 0, 1],
                [1, 0, 1],
            ],
        )

        np.testing.assert_array_equal(result, expected)


class TestGradientComputationsRelu(unittest.TestCase):
    """Tests for gradient_computations_relu."""

    def test_gradient_computations_relu_updates_gradient_dictionary_with_expected_keys(
        self,
    ) -> None:
        """Add dA, dZ, dW, and db for the current ReLU layer."""
        x = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        gradients = {
            "dZ2": np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
            ),
        }
        parameters = {
            "W2": np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ],
            ),
        }
        forward_pass_results = {
            "Z1": np.array(
                [
                    [1.0, -1.0, 0.0],
                    [-2.0, 3.0, 4.0],
                ],
            ),
        }

        result = gradients_cross_entropy.gradient_computations_relu(
            x=x,
            gradients=gradients,
            parameters=parameters,
            forward_pass_results=forward_pass_results,
            layer=1,
        )

        self.assertEqual(set(result.keys()), {"dZ2", "dA1", "dZ1", "dW1", "db1"})

    def test_gradient_computations_relu_computes_da_current(self) -> None:
        """Compute dA for the current layer from the next layer gradient."""
        x = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        gradients = {
            "dZ2": np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
            ),
        }
        parameters = {
            "W2": np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ],
            ),
        }
        forward_pass_results = {
            "Z1": np.array(
                [
                    [1.0, -1.0, 0.0],
                    [-2.0, 3.0, 4.0],
                ],
            ),
        }

        result = gradients_cross_entropy.gradient_computations_relu(
            x=x,
            gradients=gradients,
            parameters=parameters,
            forward_pass_results=forward_pass_results,
            layer=1,
        )

        expected_da = np.array(
            [
                [1.0, 2.0, 3.0],
                [3.0, 4.0, 7.0],
            ],
        )

        np.testing.assert_allclose(result["dA1"], expected_da)

    def test_gradient_computations_relu_computes_dz_current(self) -> None:
        """Compute dZ by multiplying dA by the ReLU derivative."""
        x = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        gradients = {
            "dZ2": np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
            ),
        }
        parameters = {
            "W2": np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ],
            ),
        }
        forward_pass_results = {
            "Z1": np.array(
                [
                    [1.0, -1.0, 0.0],
                    [-2.0, 3.0, 4.0],
                ],
            ),
        }

        result = gradients_cross_entropy.gradient_computations_relu(
            x=x,
            gradients=gradients,
            parameters=parameters,
            forward_pass_results=forward_pass_results,
            layer=1,
        )

        expected_dz = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 4.0, 7.0],
            ],
        )

        np.testing.assert_allclose(result["dZ1"], expected_dz)

    def test_gradient_computations_relu_computes_dw_current(self) -> None:
        """Compute dW using the previous activation and current dZ."""
        x = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        gradients = {
            "dZ2": np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
            ),
        }
        parameters = {
            "W2": np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ],
            ),
        }
        forward_pass_results = {
            "Z1": np.array(
                [
                    [1.0, -1.0, 0.0],
                    [-2.0, 3.0, 4.0],
                ],
            ),
        }

        result = gradients_cross_entropy.gradient_computations_relu(
            x=x,
            gradients=gradients,
            parameters=parameters,
            forward_pass_results=forward_pass_results,
            layer=1,
        )

        expected_dw = np.array(
            [
                [0.5, 6.0, 10.5],
                [1.0, 8.0, 14.0],
            ],
        )

        np.testing.assert_allclose(result["dW1"], expected_dw)

    def test_gradient_computations_relu_computes_db_current(self) -> None:
        """Compute db as the average dZ value per hidden neuron."""
        x = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        gradients = {
            "dZ2": np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
            ),
        }
        parameters = {
            "W2": np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ],
            ),
        }
        forward_pass_results = {
            "Z1": np.array(
                [
                    [1.0, -1.0, 0.0],
                    [-2.0, 3.0, 4.0],
                ],
            ),
        }

        result = gradients_cross_entropy.gradient_computations_relu(
            x=x,
            gradients=gradients,
            parameters=parameters,
            forward_pass_results=forward_pass_results,
            layer=1,
        )

        expected_db = np.array([[0.5, 2.0, 3.5]])

        np.testing.assert_allclose(result["db1"], expected_db)

    def test_gradient_computations_relu_uses_previous_activation_for_later_layer(
        self,
    ) -> None:
        """Use A from the previous layer when the current layer is not layer 1."""
        x = np.zeros((2, 2))
        gradients = {
            "dZ3": np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
            ),
        }
        parameters = {
            "W3": np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
            ),
        }
        forward_pass_results = {
            "A1": np.array(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ],
            ),
            "Z2": np.array(
                [
                    [1.0, -1.0],
                    [2.0, 3.0],
                ],
            ),
        }

        result = gradients_cross_entropy.gradient_computations_relu(
            x=x,
            gradients=gradients,
            parameters=parameters,
            forward_pass_results=forward_pass_results,
            layer=2,
        )

        expected_dw = np.array(
            [
                [6.5, 8.0],
                [8.5, 10.0],
                [10.5, 12.0],
            ],
        )

        np.testing.assert_allclose(result["dW2"], expected_dw)


if __name__ == "__main__":
    unittest.main()
