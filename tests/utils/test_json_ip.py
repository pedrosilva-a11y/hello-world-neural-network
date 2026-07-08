"""Tests for JSON input/output utilities."""

import tempfile
import unittest
from pathlib import Path

from utils.json_io import read_json, save_json


class TestJsonIo(unittest.TestCase):
    """Tests for JSON input/output utilities."""

    def test_save_json_creates_parent_directory_and_file(self) -> None:
        """Create missing parent directories before saving the JSON file."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            file_path = (
                Path(temporary_directory)
                / "results"
                / "experiment_1"
                / "summary.json"
            )
            data = {
                "experiment_name": "experiment_1",
                "accuracy": 0.11,
            }

            save_json(data=data, file_path=file_path)

            self.assertTrue(file_path.parent.exists())
            self.assertTrue(file_path.exists())

    def test_read_json_returns_saved_data(self) -> None:
        """Read the same data that was saved to disk."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            file_path = Path(temporary_directory) / "summary.json"
            data = {
                "experiment_name": "experiment_1",
                "loss": 29.19,
                "accuracy": 0.11,
            }

            save_json(data=data, file_path=file_path)
            result = read_json(file_path=file_path)

            self.assertEqual(result, data)

    def test_save_json_overwrites_existing_file(self) -> None:
        """Overwrite an existing JSON file when saving again."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            file_path = Path(temporary_directory) / "summary.json"

            first_data = {
                "experiment_name": "experiment_1",
                "accuracy": 0.11,
            }
            second_data = {
                "experiment_name": "experiment_1",
                "accuracy": 0.25,
            }

            save_json(data=first_data, file_path=file_path)
            save_json(data=second_data, file_path=file_path)

            result = read_json(file_path=file_path)

            self.assertEqual(result, second_data)

    def test_save_json_accepts_string_file_path(self) -> None:
        """Save JSON when file_path is provided as a string."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            file_path = str(Path(temporary_directory) / "summary.json")
            data = {
                "experiment_name": "experiment_1",
            }

            save_json(data=data, file_path=file_path)
            result = read_json(file_path=file_path)

            self.assertEqual(result, data)


if __name__ == "__main__":
    unittest.main()
