from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np
import pyrealsense2 as rs

from scripts.lib.utils import get_config_parser
from scripts.lib.utils import load_yaml_defaults

COMMAND_NAME = "extract-realsense-frames"


def parse_timestamp(hhmmss: str) -> float:
    """
    Parses a timestamp string in the format HH:MM:SS into seconds.

    Args:
        hhmmss (str): Timestamp string.

    Returns:
        float: Time in seconds.

    Raises:
        ValueError: If format is incorrect.
    """
    try:
        h, m, s = map(int, hhmmss.split(":"))
        return h * 3600 + m * 60 + s
    except Exception:
        raise ValueError(
            f"Invalid timestamp format: '{hhmmss}'. Expected format is 'HH:MM:SS'",
        )


def extract_realsense_frames(
    bag_path: str,
    output_dir: str,
    rate: float = 1.0,
    start_time: float = 0.0,
    end_time: float | None = None,
) -> None:
    """
    Extracts color frames from a RealSense .bag file at a given frame rate and within a time interval.

    Args:
        bag_path (str): Path to the RealSense .bag file.
        output_dir (str, optional): Output directory for extracted frames.
        rate (float): Frame sampling interval in seconds (e.g., 1 = every second).
        start_time (float): Starting timestamp in seconds.
        end_time (float): Ending timestamp in seconds. If None, reads until the end.
    """
    if not os.path.isfile(bag_path):
        raise FileNotFoundError(f"File not found: {bag_path}")

    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join(output_dir, "frame_*.png"))
    for file in files:
        os.remove(file)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_path, repeat_playback=False)
    config.enable_stream(rs.stream.color)

    profile = pipeline.start(config)
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)

    print(f"📦 Processing: {bag_path}")
    print(
        f"🎥 Frame rate: Every {rate:.2f}s | Time window: {start_time}s to {str(end_time) + 's' or 'EOF'}",
    )
    print(f"💾 Output dir: {output_dir}")

    frame_count = 0
    saved_count = 0
    next_capture_time = start_time

    first_timestamp = None

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            timestamp = color_frame.get_timestamp() / 1000.0  # ms to seconds

            # Normalize timestamps
            if first_timestamp is None:
                first_timestamp = timestamp

            relative_time = timestamp - first_timestamp

            if relative_time < start_time:
                continue
            if end_time is not None and relative_time > end_time:
                break
            if relative_time >= next_capture_time:
                color_image = np.asanyarray(color_frame.get_data())
                filename = os.path.join(output_dir, f"frame_{saved_count+1:05d}.png")
                cv2.imwrite(filename, color_image)
                saved_count += 1
                next_capture_time += rate

            frame_count += 1

    except RuntimeError:
        print("🛑 End of stream reached.")

    finally:
        pipeline.stop()
        print(f"✅ Saved {saved_count} frames from {frame_count} processed.")


def register_subparser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """
    Registers the 'extract-realsense-frames' subcommand.

    Args:
        subparsers (argparse._SubParsersAction): The subparser object to register with.
    """
    parser = subparsers.add_parser(
        COMMAND_NAME,
        help="Extract frames from a RealSense .bag file with timing controls.",
        parents=[get_config_parser()],
        conflict_handler="resolve",
    )
    parser.add_argument(
        "--frames",
        default=os.path.join("data", "realsense", "stream-raw.bag"),
        help="Path to the RealSense .bag file",
    )
    parser.add_argument(
        "--frames-dir",
        default=os.path.join("data", "cam4-cut-frames"),
        help="Directory to output the extracted frames",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=1,
        help="Frame extraction interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="00:00:10",
        help="Start time in HH:MM:SS format",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="00:00:27",
        help="End time in HH:MM:SS format (default: end of file)",
    )

    def wrapped(args):
        start_seconds = parse_timestamp(args.start)
        end_seconds = parse_timestamp(args.end) if args.end else None
        extract_realsense_frames(
            bag_path=args.frames,
            output_dir=args.frames_dir,
            rate=args.rate,
            start_time=start_seconds,
            end_time=end_seconds,
        )

    parser.set_defaults(func=wrapped)
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames from a RealSense .bag file.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_subparser(subparsers)
    subparser = register_subparser(subparsers)
    # Load defaults into the subparser if config is given
    args, _ = parser.parse_known_args()
    if args.config and args.command:
        load_yaml_defaults(subparser, args.config)
    args = parser.parse_args()
    args.func(args)
