"""Script for visualizing one Digit Recognizer image by index."""

from argparse import ArgumentParser
from pathlib import Path
from typing import cast

from render_entry_by_index import render_entry_by_index

from data_loading.utils.read_csv import read_csv
from preprocessing.get_combined_pixel_data import get_combined_pixel_data

ROOT_DIR = Path(__file__).resolve().parents[2]

DEFAULT_SELECTION_INDEX = 200
TRAIN_SET_PATH = ROOT_DIR / "data" / "train.csv"
TEST_SET_PATH = ROOT_DIR / "data" / "test.csv"


def parse_args() -> int:
    """Parse the target image index from the command line.

    Returns:
        Image index selected through the command-line arguments.
    """
    parser = ArgumentParser(
        description="Visualize one Digit Recognizer image by index.",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=DEFAULT_SELECTION_INDEX,
        help=f"Global image index to visualize. Default: {DEFAULT_SELECTION_INDEX}.",
    )

    args = parser.parse_args()

    return cast("int", args.index)


def main() -> None:
    """Load, combine, and render one Digit Recognizer image.

    Raises:
        ValueError: If the selected index is outside the combined dataset range.
    """
    selection_index = parse_args()

    print("Loading data...")

    df_train = read_csv(file_path=TRAIN_SET_PATH)
    df_test = read_csv(file_path=TEST_SET_PATH)

    print("Train shape:", df_train.shape)
    print("Test shape:", df_test.shape)

    print("Vertically merging sets...")
    df_combined = get_combined_pixel_data(
        df_train=df_train,
        df_test=df_test,
    )

    print("Vertically merged sets shape:", df_combined.shape)

    if selection_index < 0 or selection_index >= len(df_combined):
        raise ValueError(
            f"Invalid index {selection_index}. "
            f"Valid range is 0 to {len(df_combined) - 1}."
        )

    render_entry_by_index(
        df_pixels=df_combined,
        target_index=selection_index,
    )


if __name__ == "__main__":
    main()
