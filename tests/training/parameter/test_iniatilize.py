"""Tests for neural network parameter initialization."""

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

    def test_initialize_weights_and_bias_returns_expected_parameter_keys(self) -> None:
        """Return a dictionary containing W1 and b1 parameters."""
        result = initialize.initialize_weights_and_bias(
            x_train=self.x_train,
            h=2,
        )

        self.assertEqual(set(result.keys()), {"W1", "b1"})

    def test_initialize_weights_and_bias_returns_expected_shapes(self) -> None:
        """Initialize W1 and b1 with shapes matching input size and layer size."""
        result = initialize.initialize_weights_and_bias(
            x_train=self.x_train,
            h=2,
        )

        self.assertEqual(result["W1"].shape, (3, 2))
        self.assertEqual(result["b1"].shape, (1, 2))

    def test_initialize_weights_and_bias_initializes_bias_as_zeros(self) -> None:
        """Initialize b1 as zeros."""
        result = initialize.initialize_weights_and_bias(
            x_train=self.x_train,
            h=2,
        )

        expected_bias = np.array([[0.0, 0.0]])

        np.testing.assert_array_equal(result["b1"], expected_bias)

    def test_initialize_weights_and_bias_scales_random_weights_by_input_size(self) -> None:
        """Scale random W1 values by the inverse square root of the input size."""
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
                h=2,
            )

        expected_weights = random_values * (1 / np.sqrt(3))

        np.testing.assert_allclose(result["W1"], expected_weights)
        mock_randn.assert_called_once_with(3, 2)


if __name__ == "__main__":
    unittest.main()
