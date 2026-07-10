"""Tests for neural network parameter initialization."""

import math
import unittest
from unittest.mock import patch

import numpy as np

from training.parameter import initialize


class TestInitializeWeightsAndBias(unittest.TestCase):
    """Tests for initialize_weights_and_bias."""

    def setUp(self) -> None:
        """Create reusable input matrix test data."""
        self.x_train: np.ndarray = np.array(
            [
                [0, 255, 10],
                [100, 50, 25],
            ],
        )

    def test_initialize_weights_and_bias_returns_expected_parameter_keys_for_one_layer(
        self,
    ) -> None:
        """Return a dictionary containing W1 and b1 parameters."""
        result = initialize.initialize_weights_and_bias(
            x_train=self.x_train,
            neurons_profile=[2],
        )

        self.assertEqual(set(result.keys()), {"W1", "b1"})

    def test_initialize_weights_and_bias_returns_expected_shapes_for_one_layer(
        self,
    ) -> None:
        """Initialize W1 and b1 with shapes matching input size and layer size."""
        result = initialize.initialize_weights_and_bias(
            x_train=self.x_train,
            neurons_profile=[2],
        )

        self.assertEqual(result["W1"].shape, (3, 2))
        self.assertEqual(result["b1"].shape, (1, 2))

    def test_initialize_weights_and_bias_initializes_bias_as_zeros_for_one_layer(
        self,
    ) -> None:
        """Initialize b1 as zeros."""
        result = initialize.initialize_weights_and_bias(
            x_train=self.x_train,
            neurons_profile=[2],
        )

        expected_bias = np.array([[0.0, 0.0]])

        np.testing.assert_array_equal(result["b1"], expected_bias)

    def test_initialize_weights_and_bias_scales_random_weights_with_he_initialization(
        self,
    ) -> None:
        """Scale random W1 values using He initialization based on input size."""
        random_values = np.array(
            [
                [1.0, -1.0],
                [2.0, -2.0],
                [3.0, -3.0],
            ],
        )

        with patch.object(
            initialize.np.random,
            "randn",
            return_value=random_values,
        ) as mock_randn:
            result = initialize.initialize_weights_and_bias(
                x_train=self.x_train,
                neurons_profile=[2],
            )

        expected_weights = random_values * math.sqrt(2 / 3)

        np.testing.assert_allclose(result["W1"], expected_weights)
        mock_randn.assert_called_once_with(3, 2)

    def test_initialize_weights_and_bias_returns_expected_parameter_keys_for_two_layers(
        self,
    ) -> None:
        """Return a dictionary containing W and b parameters for two layers."""
        result = initialize.initialize_weights_and_bias(
            x_train=self.x_train,
            neurons_profile=[4, 2],
        )

        self.assertEqual(set(result.keys()), {"W1", "b1", "W2", "b2"})

    def test_initialize_weights_and_bias_returns_expected_shapes_for_two_layers(
        self,
    ) -> None:
        """Initialize parameters with shapes matching each layer transition."""
        result = initialize.initialize_weights_and_bias(
            x_train=self.x_train,
            neurons_profile=[4, 2],
        )

        self.assertEqual(result["W1"].shape, (3, 4))
        self.assertEqual(result["b1"].shape, (1, 4))

        self.assertEqual(result["W2"].shape, (4, 2))
        self.assertEqual(result["b2"].shape, (1, 2))

    def test_initialize_weights_and_bias_initializes_biases_as_zeros_for_two_layers(
        self,
    ) -> None:
        """Initialize all bias vectors as zeros."""
        result = initialize.initialize_weights_and_bias(
            x_train=self.x_train,
            neurons_profile=[4, 2],
        )

        expected_b1 = np.zeros((1, 4))
        expected_b2 = np.zeros((1, 2))

        np.testing.assert_array_equal(result["b1"], expected_b1)
        np.testing.assert_array_equal(result["b2"], expected_b2)

    def test_initialize_weights_and_bias_returns_expected_shapes_for_three_layers(
        self,
    ) -> None:
        """Initialize parameters with expected shapes for three layers."""
        result = initialize.initialize_weights_and_bias(
            x_train=self.x_train,
            neurons_profile=[4, 3, 2],
        )

        self.assertEqual(result["W1"].shape, (3, 4))
        self.assertEqual(result["b1"].shape, (1, 4))

        self.assertEqual(result["W2"].shape, (4, 3))
        self.assertEqual(result["b2"].shape, (1, 3))

        self.assertEqual(result["W3"].shape, (3, 2))
        self.assertEqual(result["b3"].shape, (1, 2))

    def test_initialize_weights_and_bias_raises_error_when_neurons_profile_is_empty(
        self,
    ) -> None:
        """Raise ValueError when neurons_profile does not contain any layer."""
        with self.assertRaisesRegex(
            ValueError,
            "neurons_profile must contain at least one layer.",
        ):
            initialize.initialize_weights_and_bias(
                x_train=self.x_train,
                neurons_profile=[],
            )


if __name__ == "__main__":
    unittest.main()
