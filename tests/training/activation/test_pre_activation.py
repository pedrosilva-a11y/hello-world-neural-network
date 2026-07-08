"""Tests for neural network linear pre-activation computation."""

import unittest

import numpy as np

from training.activation.pre_activation import compute_pre_activation


class TestComputePreActivation(unittest.TestCase):
    """Tests for compute_pre_activation."""

    def test_compute_pre_activation_returns_expected_key(self) -> None:
        """Return a dictionary containing the Z pre-activation matrix."""
        x = np.array([[1, 2]])
        w = np.array([[3], [4]])
        b = np.array([[5]])

        result = compute_pre_activation(x=x, w=w, b=b)

        self.assertEqual(set(result.keys()), {"Z"})

    def test_compute_pre_activation_computes_matrix_multiplication_plus_bias(self) -> None:
        """Compute Z as XW plus b."""
        x = np.array(
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

        result = compute_pre_activation(x=x, w=w, b=b)

        expected_z = np.array(
            [
                [32, 48],
                [59, 84],
            ],
        )

        np.testing.assert_array_equal(result["Z"], expected_z)

    def test_compute_pre_activation_supports_bias_broadcasting(self) -> None:
        """Broadcast one bias row across multiple training examples."""
        x = np.array(
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

        result = compute_pre_activation(x=x, w=w, b=b)

        expected_z = np.array(
            [
                [14, 106],
                [18, 112],
                [22, 118],
            ],
        )

        np.testing.assert_array_equal(result["Z"], expected_z)

    def test_compute_pre_activation_returns_expected_shape(self) -> None:
        """Return Z with shape equal to examples by output neurons."""
        x = np.zeros((5, 3))
        w = np.zeros((3, 4))
        b = np.zeros((1, 4))

        result = compute_pre_activation(x=x, w=w, b=b)

        self.assertEqual(result["Z"].shape, (5, 4))


if __name__ == "__main__":
    unittest.main()
