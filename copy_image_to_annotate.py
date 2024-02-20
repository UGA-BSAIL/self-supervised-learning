"""
A convenient helper script that selects and copies images
to use for active learning.
"""


import argparse
import shutil
from pathlib import Path

import yaml


def _make_parser() -> argparse.ArgumentParser:
    """
    Returns:
        The parser for CLI arguments.

    """
    parser = argparse.ArgumentParser(
        description="Copies files for active learning."
    )

    parser.add_argument(
        "-b",
        "--base-dir",
        type=Path,
        default=Path("data/05_model_input/mars_multi_camera"),
        help="The base path of the MARS image dataset.",
    )
    parser.add_argument(
        "-l",
        "--image-list",
        type=Path,
        default=Path("data/08_reporting/image_order.yml"),
        help="The path to the image ordering file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("images"),
        help="The output directory.",
    )

    parser.add_argument(
        "-s",
        "--start",
        type=int,
        default=0,
        help="The image number to start at.",
    )
    parser.add_argument(
        "-n",
        "--num-images",
        type=int,
        required=True,
        help="The number of images to copy.",
    )

    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    with open(args.image_list) as f:
        image_list = yaml.safe_load(f)

    args.output.mkdir(exist_ok=True)
    for i in range(args.start, args.start + args.num_images):
        image_path = args.base_dir / f"{image_list[i]}.jpg"
        shutil.copy(image_path, args.output / image_path.name)


if __name__ == "__main__":
    main()
