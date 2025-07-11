from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from scripts.lib.utils import get_config_parser
from scripts.lib.utils import load_camera_parameters
from scripts.lib.utils import load_yaml_defaults

COMMAND_NAME = "estimate-height"


def compute_extrinsics_from_config(config_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes extrinsic parameters (R, t) from user-friendly config.

    Args:
        config_path (str): YAML file with camera_height, pitch angle, distance_to_object.

    Returns:
        (R, t): rotation matrix (3x3), translation vector (3x1)
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    h = config["camera_height"]
    theta_deg = config["camera_pitch"]
    d = config["distance_to_object"]

    theta_rad = np.deg2rad(theta_deg)

    # Rotation around X-axis (pitch downward)
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta_rad), -np.sin(theta_rad)],
            [0, np.sin(theta_rad), np.cos(theta_rad)],
        ],
        dtype=np.float32,
    )

    # Camera translation (camera looking toward +Z, from above the ground and behind the subject)
    t = np.array([0, -h, -d], dtype=np.float32)

    return R_x, t


def backproject_to_3d(image_point, K, R, t):
    """Back-projects a 2D point to 3D assuming Z=0 (ground plane)."""
    inv_K = np.linalg.inv(K)
    ray_cam = inv_K @ np.array([image_point[0], image_point[1], 1.0])
    ray_world = R @ ray_cam
    cam_center = -R @ t.reshape(3, 1)
    s = -cam_center[2, 0] / ray_world[2]
    point_3d = cam_center.flatten() + s * ray_world
    return point_3d


def estimate_height_from_bbox(bbox, K, R, t):
    """Estimates height of a person from a single bounding box."""
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2
    foot_2d = [x_center, y_max]
    head_2d = [x_center, y_min]
    foot_3d = backproject_to_3d(foot_2d, K, R, t)
    head_3d = backproject_to_3d(head_2d, K, R, t)
    return np.linalg.norm(foot_3d - head_3d)


def detect_and_estimate(image_path, intrinsics_path, extrinsics_config, model_path):
    """Main pipeline: detect people and estimate height."""
    calib = load_camera_parameters(intrinsics_path)
    K = np.array(calib["camera_matrix"], dtype=np.float32)
    R, t = compute_extrinsics_from_config(extrinsics_config)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Run YOLO
    model = YOLO(model_path)
    results = model(image, verbose=False)[0]
    person_bboxes = [
        box
        for box, cls in zip(
            results.boxes.xyxy.cpu().numpy(),
            results.boxes.cls.cpu().numpy(),
        )
        if int(cls) == 0
    ]

    if not person_bboxes:
        print("❌ No person detected.")
        return

    # Choose tallest person (bounding box with largest height in pixels)
    bbox = max(person_bboxes, key=lambda b: b[3] - b[1])
    height = estimate_height_from_bbox(bbox, K, R, t)
    print(f"✅ Estimated height: {height:.2f} meters")


def register_subparser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Registers subparser CLI."""
    parser = subparsers.add_parser(
        COMMAND_NAME,
        help="Estimate height from an image.",
        parents=[get_config_parser()],
        conflict_handler="resolve",
    )
    parser.add_argument("--image", type=str, help="Input image path.")
    parser.add_argument(
        "--intrinsics",
        type=str,
        default=os.path.join("data", "intrinsics", "cam1.yaml"),
        help="Path to intrinsics YAML file.",
    )
    parser.add_argument(
        "--extrinsics",
        type=str,
        default=os.path.join("data", "extrinsics", "cam1.yaml"),
        help="Extrinsics config YAML file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model path.",
    )
    parser.set_defaults(
        func=lambda args: detect_and_estimate(
            args.image,
            args.intrinsics,
            args.extrinsics,
            args.model,
        ),
    )
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate person height using YOLO and camera calibration.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparser = register_subparser(subparsers)
    # Load defaults into the subparser if config is given
    args, _ = parser.parse_known_args()
    if args.config and args.command:
        load_yaml_defaults(subparser, args.config)
    args = parser.parse_args()
    args.func(args)
