"""Script for printing pixel intensity distribution."""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from pprint import pprint
from statistics.get_pixel_value_distribution import get_pixel_value_distribution

from preprocessing.get_combined_pixel_data import get_combined_pixel_data

from data_loading.utils.read_csv import read_csv

ROOT_DIR = Path(__file__).resolve().parents[2]

TRAIN_SET_PATH = ROOT_DIR / "data" / "train.csv"
TEST_SET_PATH = ROOT_DIR / "data" / "test.csv"


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = ArgumentParser(
        description="Print pixel intensity distribution for Digit Recognizer data.",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Optional image index. If omitted, counts all combined train/test pixels.",
    )

    parser.add_argument(
        "--only-observed",
        action="store_true",
        help="Only include pixel values that appear at least once.",
    )

    return parser.parse_args()


def main() -> None:
    """Load data, combine pixels, and print the pixel value distribution."""
    args = parse_args()

    print("Loading data...")

    df_train = read_csv(file_path=TRAIN_SET_PATH)
    df_test = read_csv(file_path=TEST_SET_PATH)

    print("Combining train and test pixel data...")

    df_combined = get_combined_pixel_data(
        df_train=df_train,
        df_test=df_test,
    )

    distribution = get_pixel_value_distribution(
        df_pixels=df_combined,
        target_index=args.index,
        include_missing_values=not args.only_observed,
    )

    pprint(distribution)


if __name__ == "__main__":
    main()
