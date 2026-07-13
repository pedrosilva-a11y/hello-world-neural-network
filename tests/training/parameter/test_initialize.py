"""Tests for neural network parameter initialization."""

import math
import unittest
from unittest.mock import Mock, patch

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
            random_seed=42,
        )

        self.assertEqual(set(result.keys()), {"W1", "b1"})

    def test_initialize_weights_and_bias_returns_expected_shapes_for_one_layer(
        self,
    ) -> None:
        """Initialize W1 and b1 with shapes matching input size and layer size."""
        result = initialize.initialize_weights_and_bias(
            x_train=self.x_train,
            neurons_profile=[2],
            random_seed=42,
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
            random_seed=42,
        )

        expected_bias = np.array([[0.0, 0.0]])

        np.testing.assert_array_equal(result["b1"], expected_bias)

    def test_initialize_weights_and_bias_scales_random_weights_with_he_initialization(
        self,
    ) -> None:
        """Scale standard-normal W1 values using He initialization."""
        random_values = np.array(
            [
                [1.0, -1.0],
                [2.0, -2.0],
                [3.0, -3.0],
            ],
        )
        mock_random_generator = Mock()
        mock_random_generator.standard_normal.return_value = random_values

        with patch.object(
            initialize.np.random,
            "default_rng",
            return_value=mock_random_generator,
        ) as mock_default_rng:
            result = initialize.initialize_weights_and_bias(
                x_train=self.x_train,
                neurons_profile=[2],
                random_seed=123,
            )

        expected_weights = random_values * math.sqrt(2 / 3)

        np.testing.assert_allclose(result["W1"], expected_weights)
        mock_default_rng.assert_called_once_with(123)
        mock_random_generator.standard_normal.assert_called_once_with(
            size=(3, 2),
        )

    def test_initialize_weights_and_bias_uses_none_seed_when_seed_is_not_provided(
        self,
    ) -> None:
        """Create a default random generator when random_seed is omitted."""
        random_values = np.ones((3, 2))
        mock_random_generator = Mock()
        mock_random_generator.standard_normal.return_value = random_values

        with patch.object(
            initialize.np.random,
            "default_rng",
            return_value=mock_random_generator,
        ) as mock_default_rng:
            result = initialize.initialize_weights_and_bias(
                x_train=self.x_train,
                neurons_profile=[2],
            )

        expected_weights = random_values * math.sqrt(2 / 3)

        np.testing.assert_allclose(result["W1"], expected_weights)
        mock_default_rng.assert_called_once_with(None)
        mock_random_generator.standard_normal.assert_called_once_with(
            size=(3, 2),
        )

    def test_initialize_weights_and_bias_returns_reproducible_weights_with_same_seed(
        self,
    ) -> None:
        """Return identical weight matrices when using the same seed."""
        first_result = initialize.initialize_weights_and_bias(
            x_train=self.x_train,
            neurons_profile=[4, 2],
            random_seed=42,
        )
        second_result = initialize.initialize_weights_and_bias(
            x_train=self.x_train,
            neurons_profile=[4, 2],
            random_seed=42,
        )

        np.testing.assert_array_equal(first_result["W1"], second_result["W1"])
        np.testing.assert_array_equal(first_result["W2"], second_result["W2"])
        np.testing.assert_array_equal(first_result["b1"], second_result["b1"])
        np.testing.assert_array_equal(first_result["b2"], second_result["b2"])

    def test_initialize_weights_and_bias_returns_different_weights_with_different_seeds(
        self,
    ) -> None:
        """Return different weight matrices when using different seeds."""
        first_result = initialize.initialize_weights_and_bias(
            x_train=self.x_train,
            neurons_profile=[4, 2],
            random_seed=42,
        )
        second_result = initialize.initialize_weights_and_bias(
            x_train=self.x_train,
            neurons_profile=[4, 2],
            random_seed=123,
        )

        self.assertFalse(np.array_equal(first_result["W1"], second_result["W1"]))
        self.assertFalse(np.array_equal(first_result["W2"], second_result["W2"]))

    def test_initialize_weights_and_bias_returns_expected_parameter_keys_for_two_layers(
        self,
    ) -> None:
        """Return a dictionary containing W and b parameters for two layers."""
        result = initialize.initialize_weights_and_bias(
            x_train=self.x_train,
            neurons_profile=[4, 2],
            random_seed=42,
        )

        self.assertEqual(set(result.keys()), {"W1", "b1", "W2", "b2"})

    def test_initialize_weights_and_bias_returns_expected_shapes_for_two_layers(
        self,
    ) -> None:
        """Initialize parameters with shapes matching each layer transition."""
        result = initialize.initialize_weights_and_bias(
            x_train=self.x_train,
            neurons_profile=[4, 2],
            random_seed=42,
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
            random_seed=42,
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
            random_seed=42,
        )

        self.assertEqual(result["W1"].shape, (3, 4))
        self.assertEqual(result["b1"].shape, (1, 4))

        self.assertEqual(result["W2"].shape, (4, 3))
        self.assertEqual(result["b2"].shape, (1, 3))

        self.assertEqual(result["W3"].shape, (3, 2))
        self.assertEqual(result["b3"].shape, (1, 2))

    def test_initialize_weights_and_bias_calls_random_generator_once_per_weight_matrix(
        self,
    ) -> None:
        """Create one random weight matrix per layer."""
        mock_random_generator = Mock()
        mock_random_generator.standard_normal.side_effect = [
            np.ones((3, 4)),
            np.ones((4, 3)),
            np.ones((3, 2)),
        ]

        with patch.object(
            initialize.np.random,
            "default_rng",
            return_value=mock_random_generator,
        ):
            initialize.initialize_weights_and_bias(
                x_train=self.x_train,
                neurons_profile=[4, 3, 2],
                random_seed=42,
            )

        expected_calls = [
            unittest.mock.call(size=(3, 4)),
            unittest.mock.call(size=(4, 3)),
            unittest.mock.call(size=(3, 2)),
        ]

        self.assertEqual(
            mock_random_generator.standard_normal.call_args_list,
            expected_calls,
        )

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
                random_seed=42,
            )


if __name__ == "__main__":
    unittest.main()
