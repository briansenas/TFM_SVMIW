from __future__ import annotations

import argparse
import os
from typing import Any

import cv2
from ultralytics import YOLO


def is_whole_person_in_frame(
    yolo_result: Any,
    image_height: int,
    margin: int = 10,
) -> bool:
    """
    Check if both ankles are visible and not cropped at the bottom edge.

    Args:
        yolo_result: YOLO result object containing keypoints.
        image_height: Height of the image.
        margin: Margin from the bottom to consider foot uncropped.

    Returns:
        True if both ankles are detected and well inside the image.
    """
    if not yolo_result.keypoints or len(yolo_result.keypoints.data) == 0:
        return False

    for kp in yolo_result.keypoints.data:
        left_ankle = kp[15]  # [x, y, conf]
        right_ankle = kp[16]

        if (
            left_ankle[2] > 0.5
            and right_ankle[2] > 0.5
            and left_ankle[1] < image_height - margin
            and right_ankle[1] < image_height - margin
        ):
            return True
    return False


def process_frame_directory(input_dir: str, output_dir: str, margin: int = 10) -> None:
    """
    Process a directory of image frames, saving only those where a full person
    is visible and touching the ground.

    Args:
        input_dir: Path to the input directory containing frame images.
        output_dir: Path to the output directory to save filtered frames.
    """
    # Use yolo for human detection
    yolo_model = YOLO("yolov8n-pose.pt")
    # Create the output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    # Clean up first
    image_files = sorted(
        [
            f
            for f in os.listdir(output_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ],
    )
    for image in image_files:
        os.remove(os.path.join(output_dir, image))
    image_files = sorted(
        [
            f
            for f in os.listdir(input_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ],
    )
    # Iterate over all the images
    saved_count = 0
    for _, fname in enumerate(image_files):
        img_path = os.path.join(input_dir, fname)
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        image_height, _ = frame.shape[:2]
        results = yolo_model(frame)[0]
        # If we use this with images in the wild we should add a is_upright_check
        # We should use attempt to use information regarding the ground plane (avoid climbing stuff)
        if is_whole_person_in_frame(results, image_height, margin):
            save_path = os.path.join(output_dir, fname)
            cv2.imwrite(save_path, frame)
            saved_count += 1
    print(
        f"Processed {len(image_files)} frames. Saved {saved_count} valid frames to '{output_dir}'",
    )


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    """
    Register this subcommand 'filter-frames' to the main CLI parser.

    Args:
        subparsers: Subparsers object from argparse.ArgumentParser().
    """
    parser = subparsers.add_parser(
        "filter-frames",
        help="Filter image frames where a whole person is visible",
    )
    parser.add_argument(
        "--input-dir",
        default=os.path.join("data", "original", "images"),
        help="Directory containing image frames",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("data", "processed", "images"),
        help="Directory with the filtered frames",
    )
    parser.add_argument(
        "--margin",
        default=10,
        help="Marging to tolerate against the image bottom edge",
    )
    parser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    """
    Entry point for the CLI command.

    Args:
        args: Parsed command-line arguments.
    """
    process_frame_directory(args.input_dir, args.output_dir, int(args.margin))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_subparser(subparsers)
    args = parser.parse_args()
    args.func(args)
