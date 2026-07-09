"""Tests for YAML IO utilities."""

import tempfile
import unittest
from pathlib import Path

from utils.yaml_io import read_yaml


class TestYamlIo(unittest.TestCase):
    """Tests for YAML reading utilities."""

    def test_read_yaml_reads_yaml_file_into_dictionary(self) -> None:
        """Read a valid YAML file into a dictionary."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            yaml_path = Path(temporary_directory) / "experiment.yaml"
            yaml_path.write_text(
                "\n".join(
                    [
                        "experiment_name: softmax_normalized_lr_01_5k",
                        "preprocessing:",
                        "  normalize_pixels: true",
                        "training:",
                        "  num_iterations: 5000",
                        "  learning_rate: 0.1",
                    ]
                ),
                encoding="utf-8",
            )

            result = read_yaml(file_path=yaml_path)

        expected = {
            "experiment_name": "softmax_normalized_lr_01_5k",
            "preprocessing": {
                "normalize_pixels": True,
            },
            "training": {
                "num_iterations": 5000,
                "learning_rate": 0.1,
            },
        }

        self.assertEqual(result, expected)

    def test_read_yaml_accepts_string_file_path(self) -> None:
        """Read a YAML file when the file path is provided as a string."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            yaml_path = Path(temporary_directory) / "experiment.yaml"
            yaml_path.write_text(
                "experiment_name: softmax_baseline\n",
                encoding="utf-8",
            )

            result = read_yaml(file_path=str(yaml_path))

        self.assertEqual(result, {"experiment_name": "softmax_baseline"})

    def test_read_yaml_raises_value_error_for_empty_file(self) -> None:
        """Raise ValueError when the YAML file is empty."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            yaml_path = Path(temporary_directory) / "empty.yaml"
            yaml_path.write_text("", encoding="utf-8")

            with self.assertRaises(ValueError) as context:
                read_yaml(file_path=yaml_path)

        self.assertIn("YAML file is empty", str(context.exception))

    def test_read_yaml_raises_value_error_for_non_mapping_yaml(self) -> None:
        """Raise ValueError when the YAML file does not contain a mapping."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            yaml_path = Path(temporary_directory) / "list.yaml"
            yaml_path.write_text(
                "\n".join(
                    [
                        "- softmax_baseline",
                        "- relu_h16",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError) as context:
                read_yaml(file_path=yaml_path)

        self.assertIn("YAML file must contain a top-level mapping", str(context.exception))

    def test_read_yaml_raises_file_not_found_error_for_missing_file(self) -> None:
        """Raise FileNotFoundError when the YAML file does not exist."""
        missing_path = Path("missing_experiment_config.yaml")

        with self.assertRaises(FileNotFoundError):
            read_yaml(file_path=missing_path)


if __name__ == "__main__":
    unittest.main()
