"""Tests for pixel normalization preprocessing."""

import unittest

import numpy as np

from preprocessing.normalize_pixels import normalize_pixels


class TestNormalizePixels(unittest.TestCase):
    """Tests for normalize_pixels."""

    def test_normalize_pixels_scales_values_to_zero_one_range(self) -> None:
        """It should divide raw pixel values by the default scale value."""
        x = np.array(
            [
                [0, 127.5, 255],
                [64, 128, 192],
            ]
        )

        result = normalize_pixels(x)

        expected = np.array(
            [
                [0.0, 0.5, 1.0],
                [64 / 255.0, 128 / 255.0, 192 / 255.0],
            ]
        )

        np.testing.assert_allclose(result, expected)

    def test_normalize_pixels_uses_custom_pixel_scale_value(self) -> None:
        """It should divide raw pixel values by the configured scale value."""
        x = np.array(
            [
                [0, 50, 100],
                [25, 75, 100],
            ]
        )

        result = normalize_pixels(
            x=x,
            pixel_scale_value=100.0,
        )

        expected = np.array(
            [
                [0.0, 0.5, 1.0],
                [0.25, 0.75, 1.0],
            ]
        )

        np.testing.assert_allclose(result, expected)

    def test_normalize_pixels_preserves_input_shape(self) -> None:
        """It should preserve the original matrix shape."""
        x = np.array(
            [
                [0, 255, 128],
                [64, 32, 16],
            ]
        )

        result = normalize_pixels(x)

        self.assertEqual(result.shape, x.shape)

    def test_normalize_pixels_returns_float_array(self) -> None:
        """It should return a floating-point array."""
        x = np.array([[0, 255]], dtype=np.int64)

        result = normalize_pixels(x)

        self.assertTrue(np.issubdtype(result.dtype, np.floating))

    def test_normalize_pixels_does_not_mutate_input(self) -> None:
        """It should not mutate the original input matrix."""
        x = np.array([[0, 255], [128, 64]])
        original = x.copy()

        normalize_pixels(x)

        np.testing.assert_array_equal(x, original)

    def test_normalize_pixels_handles_all_zero_matrix(self) -> None:
        """It should safely normalize an all-zero matrix."""
        x = np.zeros((2, 3), dtype=np.int64)

        result = normalize_pixels(x)

        expected = np.zeros((2, 3), dtype=float)
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
