"""Utilities for creating Kaggle Digit Recognizer submission files."""

import csv
from pathlib import Path
from typing import TypedDict

import numpy as np


class KaggleSubmissionRow(TypedDict):
    """One row in the Kaggle Digit Recognizer submission format."""

    ImageId: int
    Label: int


def create_kaggle_submission_rows(
    predictions: np.ndarray,
) -> list[KaggleSubmissionRow]:
    """Create Kaggle submission rows from model predictions.

    Args:
        predictions: One-dimensional array containing predicted digit labels.

    Returns:
        List of rows using Kaggle's expected ImageId and Label format.

    Raises:
        ValueError: If predictions is not a one-dimensional array.
    """
    if predictions.ndim != 1:
        raise ValueError("predictions must be a one-dimensional array.")

    return [
        {
            "ImageId": image_id,
            "Label": int(prediction),
        }
        for image_id, prediction in enumerate(predictions, start=1)
    ]


def save_kaggle_submission(
    predictions: np.ndarray,
    file_path: Path,
) -> None:
    """Save predictions using Kaggle's Digit Recognizer submission format.

    Args:
        predictions: One-dimensional array containing predicted digit labels.
        file_path: Destination CSV file path.

    Raises:
        ValueError: If predictions is not a one-dimensional array.
    """
    submission_rows = create_kaggle_submission_rows(predictions=predictions)

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open(mode="w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["ImageId", "Label"])

        for row in submission_rows:
            writer.writerow([row["ImageId"], row["Label"]])
