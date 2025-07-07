from __future__ import annotations

import argparse
import os
import subprocess


def cut_video(input_path: str, start_time: str, end_time: str) -> None:
    """
    Cuts a video using ffmpeg between start_time and end_time.

    Args:
        input_path (str): Path to the input video file.
        start_time (str): Start time in format HH:MM:SS or seconds (e.g., "00:01:30").
        end_time (str): End time in format HH:MM:SS or seconds (e.g., "00:02:00").
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input video file not found: {input_path}")

    # Generate output path with "-cut" suffix
    base, ext = os.path.splitext(os.path.basename(input_path))
    output_filename = f"{base}-cut{ext}"
    output_path = os.path.join(os.path.dirname(input_path), output_filename)

    # Construct FFmpeg command
    cmd = [
        "ffmpeg",
        "-i",
        input_path,
        "-ss",
        start_time,
        "-to",
        end_time,
        "-c",
        "copy",
        output_path,
    ]

    print(f"🔧 Cutting video: {input_path}")
    print(f"⏱️  From: {start_time} To: {end_time}")
    print(f"💾 Saving to: {output_path}")

    try:
        subprocess.run(cmd, check=True)
        print("✅ Video cut successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg error: {e}")


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    """
    Registers the 'cut-video' subparser command.

    Args:
        subparsers (argparse._SubParsersAction): The subparser container.
    """
    parser = subparsers.add_parser(
        "cut-video",
        help="Cut a video between two timestamps using ffmpeg.",
    )
    parser.add_argument(
        "--input",
        default=os.path.join("data", "cam1.mkv"),
        help="Path to input video.",
    )
    parser.add_argument(
        "--start",
        default="00:00:08",
        help="Start timestamp (e.g., 00:01:00).",
    )
    parser.add_argument(
        "--end",
        default="00:00:27",
        help="End timestamp (e.g., 00:02:00).",
    )
    parser.set_defaults(func=lambda args: cut_video(args.input, args.start, args.end))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cut Video Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_subparser(subparsers)
    args = parser.parse_args()
    args.func(args)
