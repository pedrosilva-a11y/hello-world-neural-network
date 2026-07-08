"""Script for visualizing pixel intensity distribution for one image."""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from pprint import pprint
from statistics.get_pixel_value_distribution import get_pixel_value_distribution

from render_pixel_value_distribution import render_pixel_value_distribution

from data_loading.utils.read_csv import read_csv
from preprocessing.get_combined_pixel_data import get_combined_pixel_data

ROOT_DIR = Path(__file__).resolve().parents[2]

DEFAULT_SELECTION_INDEX = 200
TRAIN_SET_PATH = ROOT_DIR / "data" / "train.csv"
TEST_SET_PATH = ROOT_DIR / "data" / "test.csv"


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = ArgumentParser(
        description="Visualize pixel intensity distribution for one Digit Recognizer image.",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=DEFAULT_SELECTION_INDEX,
        help=f"Image index to analyze. Default: {DEFAULT_SELECTION_INDEX}.",
    )

    parser.add_argument(
        "--exclude-zero",
        action="store_true",
        help="Exclude pixel value 0 from the printed distribution and plot.",
    )

    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use logarithmic scale on the y-axis.",
    )

    return parser.parse_args()


def main() -> None:
    """Load data, print image-level pixel distribution, and render a bar chart."""
    args = parse_args()

    print(f"Loading data for image index: {args.index}")

    df_train = read_csv(file_path=TRAIN_SET_PATH)
    df_test = read_csv(file_path=TEST_SET_PATH)

    df_combined = get_combined_pixel_data(
        df_train=df_train,
        df_test=df_test,
    )

    distribution = get_pixel_value_distribution(
        df_pixels=df_combined,
        target_index=args.index,
    )

    if args.exclude_zero:
        distribution.pop(0, None)

    print(f"Pixel value distribution for image index {args.index}:")
    pprint(distribution)

    render_pixel_value_distribution(
        distribution=distribution,
        title=f"Pixel intensity distribution for image index {args.index}",
        use_log_scale=args.log_scale,
    )


if __name__ == "__main__":
    main()
