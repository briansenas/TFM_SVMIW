from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import tqdm
from ultralytics import YOLO

from scripts.lib.utils import get_config_parser
from scripts.lib.utils import load_camera_parameters
from scripts.lib.utils import load_yaml_defaults

COMMAND_NAME = "estimate-height"


def build_rotation_matrix(angle_x_deg, angle_y_deg):
    """
    Computes the rotation matrix given 2 angles in degree value
    Args:
        angle_x_deg (float): The degree in the X axis
        angle_y_deg (float): The degree in the Y axis

    Returns:
        R: rotation matrix (3x3)
    """
    # Convert to radians
    ax = np.deg2rad(angle_x_deg)
    ay = np.deg2rad(angle_y_deg)
    # Rotation around X (pitch)
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]],
    )
    # Rotation around Y (yaw)
    Ry = np.array(
        [[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]],
    )
    # Order matters
    R = Ry @ Rx
    return R


def backproject_to_3d(image_point, K, R, T, s):
    """Back-projects a 2D point to 3D assumes Z=0"""
    # Converts the 2D pixel coordinate (image_point) into a 3D ray in the camera's local coordinate system
    inv_K = np.linalg.inv(K)
    ray_cam = inv_K @ np.array([image_point[0], image_point[1], 1.0])
    # Rotates the camera ray using the rotation matrix to align it with the world coordinate system.
    ray_world = R @ ray_cam
    # Calculates the camera's physical position (center) in the world.
    cam_center = -R.T @ T.reshape(3, 1)
    # Scaling factor
    if s is None:
        s = -cam_center[2, 0] / ray_world[2]
    # Travel from center to the object
    point_3d = cam_center.flatten() + s * ray_world
    return point_3d, s


def estimate_height_from_bbox(bbox, K, R, T, DC):
    """Estimates height of a person from a single bounding box."""
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2
    foot_2d = [x_center, y_max]
    head_2d = [x_center, y_min]
    foot_3d, s = backproject_to_3d(foot_2d, K, R, T, None)
    head_3d, s = backproject_to_3d(head_2d, K, R, T, s)
    return np.linalg.norm(foot_3d - head_3d)


def get_camera_matrix(intrinsics_path):
    calib = load_camera_parameters(intrinsics_path)
    K = np.array(calib["camera_matrix"], dtype=np.float32)
    DC = np.array(calib["dist_coeffs"], dtype=np.float32)
    return K, DC


def read_camera_parameters(data: dict):
    K, DC = get_camera_matrix(data["intrinsics"])
    h = data["camera_height"]
    theta_deg = data["camera_pitch"]
    yaw_deg = data.get("camera_yaw", 0.0)
    d = data["distance"]
    R = build_rotation_matrix(theta_deg, yaw_deg)
    # Camera translation (camera looking toward +Z, from above the ground and behind the subject)
    # Basically translating the camera down (to the ground), and towards the object
    T = np.array([0, h, -d], dtype=np.float32)
    return K, R, T, DC


def batch_detect_and_estimate(
    data: list[cv2.typing.MatLike],
    model: YOLO | None = None,
):
    images = list(map(lambda x: cv2.imread(x["image_path"]), data))
    extrinsics = list(map(read_camera_parameters, data))
    if model is not None:
        results = model(images, verbose=False)
        for i, result in enumerate(results):
            person_bboxes = [
                box
                for box, cls, conf in zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.cls.cpu().numpy(),
                    result.boxes.conf.cpu().numpy(),
                )
                if int(cls) == 0 and float(conf) > 0.75
            ]
        # Choose tallest person (bounding box with largest height in pixels)
        bbox = max(person_bboxes, key=lambda b: b[3] - b[1])
        height = estimate_height_from_bbox(bbox, *extrinsics[i])
        data[i]["pred_height"] = height
    else:
        for i, sample in enumerate(data):
            bbox = sample["gt_bbox"]
            height = estimate_height_from_bbox(bbox, *extrinsics[i])
            data[i]["pred_height"] = height
    return data


def main(args: argparse.Namespace):
    with open(args.input_json, encoding="utf-8") as file:
        data = json.load(file)
    if not data:
        raise ValueError("The image list is empty")
    batch_size = args.batch_size
    detect(
        data=data,
        model_name=args.model,
        batch_size=batch_size,
        output_file=args.output_file,
    )


def detect(
    data: list[dict],
    output_file: str | None = None,
    model_name: str = "yolov8n.pt",
    batch_size: int = 32,
):
    """
    Detects and estimates the height of images given extrinsics and intrinsics
    Args:
        data (list[dict]): Dictionary with image path and extrinsics
        model (str): Yolo model name
        batch_size (int): The amount of images per batch for the model
        output_file (str): The path to save the predictions
    Returns:
        dict
    """
    print(f"Preparing to process {len(data)} images")
    if "gt_bbox" not in data[0]:
        model = YOLO(model_name)
    else:
        model = None
    image_array = np.array_split(
        np.asarray(data),
        max(len(data) // batch_size, 1),
    )
    predictions = []
    for batch in tqdm.tqdm(image_array):
        predictions.extend(
            batch_detect_and_estimate(
                data=batch.tolist(),
                model=model,
            ),
        )
    if output_file:
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(json.dumps(data))
    return data


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
    parser.add_argument(
        "--input-json",
        type=str,
        help="JSON with images path for batch detection",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Extrinsics config YAML file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/mydataset-estimation.json",
        help="Extrinsics config YAML file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model path.",
    )

    parser.set_defaults(func=main)
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
