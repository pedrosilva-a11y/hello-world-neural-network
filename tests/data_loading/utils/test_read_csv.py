"""Tests for CSV loading utilities."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from data_loading.utils.read_csv import read_csv


class TestReadCsv(unittest.TestCase):
    """Tests for the read_csv utility function."""

    def test_read_csv_with_path_returns_expected_dataframe(self) -> None:
        """Read a CSV from a Path object and verify the returned DataFrame."""
        with TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "sample.csv"
            csv_path.write_text(
                "label,pixel0,pixel1\n"
                "7,0,255\n"
                "2,10,120\n",
                encoding="utf-8",
            )

            result = read_csv(file_path=csv_path)

            expected = pd.DataFrame(
                {
                    "label": [7, 2],
                    "pixel0": [0, 10],
                    "pixel1": [255, 120],
                }
            )

            self.assertIsInstance(result, pd.DataFrame)
            pd.testing.assert_frame_equal(result, expected)

    def test_read_csv_with_string_path_returns_expected_dataframe(self) -> None:
        """Read a CSV from a string path and verify the returned DataFrame."""
        with TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "sample.csv"
            csv_path.write_text(
                "label,pixel0,pixel1\n"
                "7,0,255\n"
                "2,10,120\n",
                encoding="utf-8",
            )

            result = read_csv(file_path=str(csv_path))

            expected = pd.DataFrame(
                {
                    "label": [7, 2],
                    "pixel0": [0, 10],
                    "pixel1": [255, 120],
                }
            )

            self.assertIsInstance(result, pd.DataFrame)
            pd.testing.assert_frame_equal(result, expected)

    def test_read_csv_raises_file_not_found_error_for_missing_file(self) -> None:
        """Raise FileNotFoundError when the CSV file does not exist."""
        missing_path = Path("missing-file.csv")

        with self.assertRaises(FileNotFoundError):
            read_csv(file_path=missing_path)


if __name__ == "__main__":
    unittest.main()
