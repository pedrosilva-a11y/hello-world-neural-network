"""Tests for Digit Recognizer pixel intensity statistics."""

import unittest
from statistics.get_pixel_value_distribution import get_pixel_value_distribution

import pandas as pd


class TestGetPixelValueDistribution(unittest.TestCase):
    """Tests for pixel value distribution calculations."""

    def test_returns_image_level_distribution_with_only_observed_values(self) -> None:
        """Count pixel intensity values for one selected image."""
        df_pixels = pd.DataFrame(
            {
                "pixel0": [0],
                "pixel1": [0],
                "pixel2": [128],
                "pixel3": [255],
            }
        )

        result = get_pixel_value_distribution(
            df_pixels=df_pixels,
            target_index=0,
            include_missing_values=False,
        )

        expected = {
            0: 2,
            128: 1,
            255: 1,
        }

        self.assertEqual(result, expected)

    def test_returns_full_dataset_distribution_with_only_observed_values(self) -> None:
        """Count pixel intensity values across all rows when no index is selected."""
        df_pixels = pd.DataFrame(
            {
                "pixel0": [0, 0],
                "pixel1": [255, 128],
            }
        )

        result = get_pixel_value_distribution(
            df_pixels=df_pixels,
            include_missing_values=False,
        )

        expected = {
            0: 2,
            128: 1,
            255: 1,
        }

        self.assertEqual(result, expected)

    def test_includes_all_pixel_values_when_include_missing_values_is_true(self) -> None:
        """Return keys from 0 to 255 when missing values are included."""
        df_pixels = pd.DataFrame(
            {
                "pixel0": [0],
                "pixel1": [255],
            }
        )

        result = get_pixel_value_distribution(
            df_pixels=df_pixels,
            target_index=0,
            include_missing_values=True,
        )

        self.assertEqual(len(result), 256)
        self.assertEqual(list(result.keys()), list(range(256)))
        self.assertEqual(result[0], 1)
        self.assertEqual(result[1], 0)
        self.assertEqual(result[254], 0)
        self.assertEqual(result[255], 1)

    def test_raises_value_error_for_empty_dataframe(self) -> None:
        """Raise ValueError when the pixel DataFrame is empty."""
        df_pixels = pd.DataFrame()

        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            get_pixel_value_distribution(df_pixels=df_pixels)

    def test_raises_value_error_for_negative_index(self) -> None:
        """Raise ValueError when the selected image index is negative."""
        df_pixels = pd.DataFrame(
            {
                "pixel0": [0],
                "pixel1": [255],
            }
        )

        with self.assertRaisesRegex(ValueError, "Invalid index"):
            get_pixel_value_distribution(df_pixels=df_pixels, target_index=-1)

    def test_raises_value_error_for_out_of_range_index(self) -> None:
        """Raise ValueError when the selected image index is out of range."""
        df_pixels = pd.DataFrame(
            {
                "pixel0": [0],
                "pixel1": [255],
            }
        )

        with self.assertRaisesRegex(ValueError, "Invalid index"):
            get_pixel_value_distribution(df_pixels=df_pixels, target_index=1)

    def test_raises_value_error_when_pixel_value_is_below_zero(self) -> None:
        """Raise ValueError when a pixel value is below the valid range."""
        df_pixels = pd.DataFrame(
            {
                "pixel0": [-1],
                "pixel1": [255],
            }
        )

        with self.assertRaisesRegex(ValueError, "between 0 and 255"):
            get_pixel_value_distribution(df_pixels=df_pixels)

    def test_raises_value_error_when_pixel_value_is_above_255(self) -> None:
        """Raise ValueError when a pixel value is above the valid range."""
        df_pixels = pd.DataFrame(
            {
                "pixel0": [0],
                "pixel1": [256],
            }
        )

        with self.assertRaisesRegex(ValueError, "between 0 and 255"):
            get_pixel_value_distribution(df_pixels=df_pixels)


if __name__ == "__main__":
    unittest.main()
