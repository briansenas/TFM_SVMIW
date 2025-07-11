from __future__ import annotations

import argparse
import glob
import os
import subprocess

from scripts.lib.utils import get_config_parser
from scripts.lib.utils import load_yaml_defaults

COMMAND_NAME = "extract-frames"


def extract_frames(input_path: str, output_dir: str, rate: float) -> None:
    """
    Extracts frames from a video file using ffmpeg at a specified frame rate.

    Args:
        input_path (str): Path to the input video file.
        rate (float): Frame extraction rate (frames per second).
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join(output_dir, "frame_*.png"))
    for file in files:
        os.remove(file)

    output_pattern = os.path.join(output_dir, "frame_%05d.png")
    command = [
        "ffmpeg",
        "-i",
        input_path,
        "-vf",
        f"fps={rate}",
        output_pattern,
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Frames extracted to: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg execution: {e}")
        raise


def register_subparser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """
    Registers the 'extract-frames' subparser for CLI.

    Args:
        subparsers (argparse._SubParsersAction): The subparsers object from the main parser.
    """
    parser = subparsers.add_parser(
        COMMAND_NAME,
        help="Extract frames from a video at a specified frame rate.",
        parents=[get_config_parser()],
        conflict_handler="resolve",
    )
    parser.add_argument(
        "--frames",
        type=str,
        default=os.path.join("data", "cam1-cut.mkv"),
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        default=os.path.join("data", "cam1-cut-frames"),
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=1,
        help="Frame extraction rate (frames per second).",
    )
    parser.set_defaults(
        func=lambda args: extract_frames(args.frames, args.frames_dir, args.rate),
    )
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparser = register_subparser(subparsers)
    # Load defaults into the subparser if config is given
    args, _ = parser.parse_known_args()
    if args.config and args.command:
        load_yaml_defaults(subparser, args.config)
    args = parser.parse_args()
    args.func(args)
