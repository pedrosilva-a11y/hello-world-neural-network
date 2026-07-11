"""Tests for ReLU activation."""

import unittest

import numpy as np

from training.activation.relu import relu


class TestRelu(unittest.TestCase):
    """Tests for relu."""

    def test_relu_preserves_positive_values(self) -> None:
        """Preserve positive values."""
        pre_activation = np.array([1.0, 2.5, 10.0])

        result = relu(pre_activation)

        expected = np.array([1.0, 2.5, 10.0])

        np.testing.assert_array_equal(result, expected)

    def test_relu_replaces_negative_values_with_zero(self) -> None:
        """Replace negative values with zero."""
        pre_activation = np.array([-1.0, -2.5, -10.0])

        result = relu(pre_activation)

        expected = np.array([0.0, 0.0, 0.0])

        np.testing.assert_array_equal(result, expected)

    def test_relu_preserves_zero_values(self) -> None:
        """Preserve zero values."""
        pre_activation = np.array([0.0, 0.0, 0.0])

        result = relu(pre_activation)

        expected = np.array([0.0, 0.0, 0.0])

        np.testing.assert_array_equal(result, expected)

    def test_relu_applies_element_wise_to_mixed_values(self) -> None:
        """Apply ReLU element-wise to mixed positive, negative, and zero values."""
        pre_activation = np.array([-2.0, -0.5, 0.0, 1.5, 3.0])

        result = relu(pre_activation)

        expected = np.array([0.0, 0.0, 0.0, 1.5, 3.0])

        np.testing.assert_array_equal(result, expected)

    def test_relu_preserves_input_shape_for_matrix(self) -> None:
        """Preserve the shape of a two-dimensional input array."""
        pre_activation = np.array(
            [
                [-1.0, 0.0, 2.0],
                [3.0, -4.0, 5.0],
            ],
        )

        result = relu(pre_activation)

        expected = np.array(
            [
                [0.0, 0.0, 2.0],
                [3.0, 0.0, 5.0],
            ],
        )

        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.shape, pre_activation.shape)


if __name__ == "__main__":
    unittest.main()
