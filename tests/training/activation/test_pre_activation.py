"""Tests for neural network linear pre-activation computation."""

import unittest

import numpy as np

from training.activation.pre_activation import compute_pre_activation


class TestComputePreActivation(unittest.TestCase):
    """Tests for compute_pre_activation."""

    def test_compute_pre_activation_returns_expected_layer_key(self) -> None:
        """Return a dictionary containing the layer-specific Z key."""
        a = np.array([[1, 2]])
        w = np.array([[3], [4]])
        b = np.array([[5]])

        result = compute_pre_activation(
            a=a,
            w=w,
            b=b,
            layer_number=1,
        )

        self.assertEqual(set(result.keys()), {"Z1"})

    def test_compute_pre_activation_uses_layer_number_in_output_key(self) -> None:
        """Return Z key matching the provided layer number."""
        a = np.array([[1, 2]])
        w = np.array([[3], [4]])
        b = np.array([[5]])

        result = compute_pre_activation(
            a=a,
            w=w,
            b=b,
            layer_number=2,
        )

        self.assertEqual(set(result.keys()), {"Z2"})

    def test_compute_pre_activation_computes_matrix_multiplication_plus_bias(
        self,
    ) -> None:
        """Compute Z as previous activation times W plus b."""
        a = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
        )
        w = np.array(
            [
                [1, 2],
                [3, 4],
                [5, 6],
            ],
        )
        b = np.array([[10, 20]])

        result = compute_pre_activation(
            a=a,
            w=w,
            b=b,
            layer_number=1,
        )

        expected_z = np.array(
            [
                [32, 48],
                [59, 84],
            ],
        )

        np.testing.assert_array_equal(result["Z1"], expected_z)

    def test_compute_pre_activation_supports_bias_broadcasting(self) -> None:
        """Broadcast one bias row across multiple training examples."""
        a = np.array(
            [
                [1, 1],
                [2, 2],
                [3, 3],
            ],
        )
        w = np.array(
            [
                [1, 2],
                [3, 4],
            ],
        )
        b = np.array([[10, 100]])

        result = compute_pre_activation(
            a=a,
            w=w,
            b=b,
            layer_number=1,
        )

        expected_z = np.array(
            [
                [14, 106],
                [18, 112],
                [22, 118],
            ],
        )

        np.testing.assert_array_equal(result["Z1"], expected_z)

    def test_compute_pre_activation_returns_expected_shape(self) -> None:
        """Return Z with shape equal to examples by current layer neurons."""
        a = np.zeros((5, 3))
        w = np.zeros((3, 4))
        b = np.zeros((1, 4))

        result = compute_pre_activation(
            a=a,
            w=w,
            b=b,
            layer_number=1,
        )

        self.assertEqual(result["Z1"].shape, (5, 4))


if __name__ == "__main__":
    unittest.main()
