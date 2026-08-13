from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from scripts.lib.utils import get_config_parser

LOGGER = logging.getLogger(__file__)

COMMAND_NAME = "filter-frames"

LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6


def straight_knees_below_hips(keypoints, threshold=0.2):
    # Torso length for normalization
    shoulder = (keypoints[LEFT_SHOULDER, :2] + keypoints[RIGHT_SHOULDER, :2]) / 2

    hip_center = (keypoints[LEFT_HIP, :2] + keypoints[RIGHT_HIP, :2]) / 2

    torso_length = np.linalg.norm((shoulder - hip_center).cpu())

    left_dx = abs(keypoints[LEFT_KNEE, 0] - keypoints[LEFT_HIP, 0])
    right_dx = abs(keypoints[RIGHT_KNEE, 0] - keypoints[RIGHT_HIP, 0])

    return int(
        (
            left_dx / torso_length < threshold
            and right_dx / torso_length < threshold
            and keypoints[LEFT_KNEE, 1] > keypoints[LEFT_HIP, 1]
            and keypoints[RIGHT_KNEE, 1] > keypoints[RIGHT_HIP, 1]
        )
        .cpu()
        .item(),
    )


def torso_angle(keypoints):
    """
    keypoints: dict with
        left_shoulder
        right_shoulder
        left_hip
        right_hip
    """
    shoulder = (keypoints[LEFT_SHOULDER, :2] + keypoints[RIGHT_SHOULDER, :2]) / 2

    hip = (keypoints[LEFT_HIP, :2] + keypoints[RIGHT_HIP, :2]) / 2

    torso = (shoulder - hip).cpu()
    torso = torso / np.linalg.norm(torso)
    vertical = np.array([0.0, -1.0])
    angle = np.degrees(
        np.arccos(
            np.clip(np.dot(torso, vertical), -1, 1),
        ),
    )
    return int(angle < 15)


def is_whole_person_in_frame(
    keypoints: Any,
    image_height: int,
    image_width: int,
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
    left_ankle = keypoints[LEFT_ANKLE]  # [x, y, conf]
    right_ankle = keypoints[RIGHT_ANKLE]

    return (
        (
            left_ankle[2] > 0.5
            and right_ankle[2] > 0.5
            and left_ankle[1] < image_height - margin
            and right_ankle[1] < image_height - margin
            and left_ankle[0] < image_width - margin
            and right_ankle[0] < image_width - margin
            and left_ankle[0] > margin
            and right_ankle[0] > margin
        )
        .cpu()
        .item()
    )


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
    image_files = sorted(
        [
            f
            for f in os.listdir(input_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ],
    )
    # Iterate over all the images
    valid_imgs_json = []
    invalid_imgs_json = []
    for _, fname in enumerate(image_files):
        img_path = os.path.join(input_dir, fname)
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        image_height, image_width = frame.shape[:2]
        results = yolo_model(frame)[0]
        if not results.keypoints or len(results.keypoints.data) == 0:
            invalid_imgs_json.append(
                {
                    "img_path": img_path,
                    "is_visible": False,
                    "is_upright": False,
                    "is_knee_straight": False,
                    "no_keypoints": True,
                    "too_many_detections": False,
                },
            )
            continue
        if len(results.keypoints.data) > 1:
            invalid_imgs_json.append(
                {
                    "img_path": img_path,
                    "is_visible": False,
                    "is_upright": False,
                    "is_knee_straight": False,
                    "no_keypoints": False,
                    "too_many_detections": True,
                },
            )
            continue

        for kp in results.keypoints.data:
            is_visible = is_whole_person_in_frame(
                kp,
                image_height=image_height,
                image_width=image_width,
                margin=margin,
            )
            is_upright = torso_angle(kp)
            is_knee_straight = straight_knees_below_hips(kp)
            if is_visible and is_upright:
                valid_imgs_json.append(img_path)
            else:
                invalid_imgs_json.append(
                    {
                        "img_path": img_path,
                        "is_visible": is_visible,
                        "is_upright": is_upright,
                        "is_knee_straight": is_knee_straight,
                        "no_keypoints": False,
                        "too_many_detections": False,
                    },
                )
    with open(
        os.path.join(output_dir, "valid_filtered_imgs.json"),
        "w",
        encoding="utf-8",
    ) as file:
        file.write(json.dumps(valid_imgs_json))
    with open(
        os.path.join(output_dir, "invalid_filtered_imgs.json"),
        "w",
        encoding="utf-8",
    ) as file:
        file.write(json.dumps(invalid_imgs_json))
    print(
        f"Processed {len(image_files)} frames. Saved {len(valid_imgs_json)} valid frames to '{output_dir}'",
    )


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    """
    Register this subcommand 'filter-frames' to the main CLI parser.

    Args:
        subparsers: Subparsers object from argparse.ArgumentParser().
    """
    parser = subparsers.add_parser(
        COMMAND_NAME,
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
    return parser


def main(args: argparse.Namespace) -> None:
    """
    Entry point for the CLI command.

    Args:
        args: Parsed command-line arguments.
    """
    process_frame_directory(args.input_dir, args.output_dir, int(args.margin))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_config_parser()])
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_subparser(subparsers)
    args = parser.parse_args()
    args.func(args)
