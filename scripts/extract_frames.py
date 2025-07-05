from __future__ import annotations

import argparse
import glob
import os
import subprocess


def extract_frames(input_path: str, rate: float) -> None:
    """
    Extracts frames from a video file using ffmpeg at a specified frame rate.

    Args:
        input_path (str): Path to the input video file.
        rate (float): Frame extraction rate (frames per second).
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    base_dir = os.path.dirname(os.path.abspath(input_path))
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = os.path.join(base_dir, f"{base_name}_frames")

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


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    """
    Registers the 'extract-frames' subparser for CLI.

    Args:
        subparsers (argparse._SubParsersAction): The subparsers object from the main parser.
    """
    parser = subparsers.add_parser(
        "extract-frames",
        help="Extract frames from a video at a specified frame rate.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join("data", "cam1-cut.mkv"),
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=1,
        help="Frame extraction rate (frames per second).",
    )
    parser.set_defaults(func=lambda args: extract_frames(args.input, args.rate))
