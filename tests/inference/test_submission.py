"""Tests for Kaggle submission utilities."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from inference.submission import (
    create_kaggle_submission_rows,
    save_kaggle_submission,
)


class TestCreateKaggleSubmissionRows(unittest.TestCase):
    """Tests for create_kaggle_submission_rows."""

    def test_create_kaggle_submission_rows_uses_one_indexed_image_ids(self) -> None:
        """Create rows with ImageId starting at one."""
        predictions = np.array([2, 0, 9])

        result = create_kaggle_submission_rows(predictions=predictions)

        self.assertEqual(
            result,
            [
                {"ImageId": 1, "Label": 2},
                {"ImageId": 2, "Label": 0},
                {"ImageId": 3, "Label": 9},
            ],
        )

    def test_create_kaggle_submission_rows_converts_numpy_ints_to_python_ints(
        self,
    ) -> None:
        """Convert NumPy integer labels into plain Python integers."""
        predictions = np.array([1, 8], dtype=np.int64)

        result = create_kaggle_submission_rows(predictions=predictions)

        self.assertIsInstance(result[0]["ImageId"], int)
        self.assertIsInstance(result[0]["Label"], int)
        self.assertIsInstance(result[1]["ImageId"], int)
        self.assertIsInstance(result[1]["Label"], int)

    def test_create_kaggle_submission_rows_raises_error_for_non_1d_predictions(
        self,
    ) -> None:
        """Raise ValueError when predictions is not one-dimensional."""
        predictions = np.array(
            [
                [1, 2],
                [3, 4],
            ],
        )

        with self.assertRaisesRegex(
            ValueError,
            "predictions must be a one-dimensional array.",
        ):
            create_kaggle_submission_rows(predictions=predictions)


class TestSaveKaggleSubmission(unittest.TestCase):
    """Tests for save_kaggle_submission."""

    def test_save_kaggle_submission_creates_parent_directory_and_csv_file(self) -> None:
        """Create parent directories and save the submission CSV."""
        predictions = np.array([2, 0, 9])

        with tempfile.TemporaryDirectory() as temporary_directory:
            file_path = (
                Path(temporary_directory)
                / "nested"
                / "submission"
                / "submission.csv"
            )

            save_kaggle_submission(
                predictions=predictions,
                file_path=file_path,
            )

            self.assertTrue(file_path.exists())
            self.assertTrue(file_path.is_file())

            with file_path.open(mode="r", encoding="utf-8", newline="") as csv_file:
                reader = csv.reader(csv_file)
                rows = list(reader)

        self.assertEqual(
            rows,
            [
                ["ImageId", "Label"],
                ["1", "2"],
                ["2", "0"],
                ["3", "9"],
            ],
        )

    def test_save_kaggle_submission_raises_error_for_non_1d_predictions(self) -> None:
        """Raise ValueError when saving non-one-dimensional predictions."""
        predictions = np.array(
            [
                [1, 2],
                [3, 4],
            ],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            file_path = Path(temporary_directory) / "submission.csv"

            with self.assertRaisesRegex(
                ValueError,
                "predictions must be a one-dimensional array.",
            ):
                save_kaggle_submission(
                    predictions=predictions,
                    file_path=file_path,
                )


if __name__ == "__main__":
    unittest.main()
