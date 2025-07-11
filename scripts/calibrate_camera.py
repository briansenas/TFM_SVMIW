from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from scripts.lib.utils import get_config_parser
from scripts.lib.utils import load_yaml_defaults

COMMAND_NAME = "calibrate-camera"


def save_calibration_yaml(
    output_file: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
    sensor_size_mm: tuple[float, float],
    skew: float = 0.0,
):
    """
    Saves OpenCV calibration results into a structured YAML file.

    The format matches:
      distortion_coefficients:
        radial:
          k1: …
          k2: …
          k3: …
          k4: …
          k5: …
          k6: …
        tangential:
          p1: …
          p2: …
      focal_length_mm:
        fx: …
        fy: …
      focal_length_pixels:
        fx: …
        fy: …
      principal_point_mm:
        cx: …
        cy: …
      principal_point_pixels:
        cx: …
        cy: …
      skew: …

    Args:
        output_file (str): Path to write the YAML file.
        camera_matrix (np.ndarray): 3×3 matrix from calibrateCamera.
        dist_coeffs (np.ndarray): Distortion vector (k1,k2,p1,p2,k3…).
        image_size (tuple[int,int]): (width_px, height_px).
        sensor_size_mm (tuple[float,float]): (sensor_width_mm, sensor_height_mm).
        skew (float): Skew coefficient (usually 0).
    """
    fx_px, fy_px = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx_px, cy_px = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    sensor_w_mm, sensor_h_mm = sensor_size_mm
    img_w, img_h = image_size

    # Compute focal lengths in mm
    fx_mm = float(fx_px * (sensor_w_mm / img_w))
    fy_mm = float(fy_px * (sensor_h_mm / img_h))

    # Distortion: radial = k1,k2,k3,k4,k5,k6; tangential = p1,p2
    dc = dist_coeffs.flatten()
    radial_keys = ["k1", "k2", "k3", "k4", "k5", "k6"]
    tangential_keys = ["p1", "p2"]
    radial_vals = {
        k: float(dc[i]) if i < len(dc) else 0.0 for i, k in enumerate(radial_keys)
    }
    tangential_vals = {
        k: float(dc[2 + i]) if 2 + i < len(dc) else 0.0
        for i, k in enumerate(tangential_keys)
    }

    data = {
        "distortion_coefficients": {
            "radial": radial_vals,
            "tangential": tangential_vals,
        },
        "focal_length_mm": {"fx": fx_mm, "fy": fy_mm},
        "focal_length_pixels": {"fx": fx_px, "fy": fy_px},
        "principal_point_mm": {
            "cx": cx_px * (sensor_w_mm / img_w),
            "cy": cy_px * (sensor_h_mm / img_h),
        },
        "principal_point_pixels": {"cx": cx_px, "cy": cy_px},
        "skew": float(skew),
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False)


def calibrate_camera(
    input_dir: str,
    board_width: int,
    board_height: int,
    square_size: float,
    fisheye: bool,
    output_file: str,
    sensor_size: tuple[float, float],
) -> None:
    """
    Calibrates a camera using images of a chessboard pattern.

    Args:
        input_dir (str): Directory containing calibration images (e.g., extracted frames).
        board_width (int): Number of inner corners per chessboard row.
        board_height (int): Number of inner corners per chessboard column.
        square_size (float): Size of the chessboard squares in meters
        fisheye (bool): Flag that tells which camera model to use.
        output_file (str): Directory where the calibration parameters will be saved.
    """
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))

    if not image_paths:
        raise FileNotFoundError("No PNG images found in the input directory.")

    # Prepare object points (0,0,0), (1,0,0), ..., (width-1,height-1,0)
    objp = np.zeros((board_height * board_width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_width, 0:board_height].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    for image_path in tqdm(image_paths, desc="Processing frames"):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(
            gray,
            (board_width, board_height),
            None,
        )
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria=(
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                ),
            )
            imgpoints.append(corners2)

    if not objpoints:
        raise RuntimeError("No valid chessboard patterns were found in the images.")

    image_size = gray.shape[::-1]
    if not fisheye:
        ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
            objpoints,
            imgpoints,
            image_size,
            None,
            None,
        )
    else:
        camera_matrix = np.zeros((3, 3))
        dist_coeffs = np.zeros((4, 1))
        rvecs = []  # type: ignore
        tvecs = []  # type: ignore
        image_size = gray.shape[::-1]
        objpoints = np.expand_dims(np.asarray(objpoints), -2)
        ret, _, _, _, _ = cv2.fisheye.calibrate(
            np.asarray(objpoints),
            imgpoints,
            image_size,
            camera_matrix,
            dist_coeffs,
            rvecs,
            tvecs,
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            + cv2.fisheye.CALIB_CHECK_COND
            + cv2.fisheye.CALIB_FIX_SKEW,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        )

    save_calibration_yaml(
        output_file,
        camera_matrix,
        dist_coeffs,
        image_size=image_size,
        sensor_size_mm=sensor_size,
        skew=camera_matrix[0, 1],
    )

    print(f"Camera calibration saved to: {output_file}")


def register_subparser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """
    Registers the 'calibrate-camera' subparser for CLI.

    Args:
        subparsers (argparse._SubParsersAction): The subparsers object from the main parser.
    """
    parser = subparsers.add_parser(
        COMMAND_NAME,
        help="Calibrate a camera using extracted chessboard images.",
        parents=[get_config_parser()],
        conflict_handler="resolve",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join("data", "cam1__stream_rgb_frames"),
        help="Path to the input directory with chessboard images.",
    )
    parser.add_argument(
        "--board-width",
        type=int,
        default=9,
        help="Number of inner corners per chessboard row.",
    )
    parser.add_argument(
        "--board-height",
        type=int,
        default=6,
        help="Number of inner corners per chessboard column.",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=0.025,
        help="Size of the chessboard squares in meters",
    )
    parser.add_argument(
        "--sensor-width",
        type=float,
        default=5.7,
        help="Width of the sensor",
    )
    parser.add_argument(
        "--sensor-height",
        type=float,
        default=3,
        help="Height of the sensor",
    )
    parser.add_argument(
        "--fisheye",
        action="store_true",
        help="Flag to determine whether to the camera has a fisheye model.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("data", "intrinsics", "cam1.yaml"),
        help="Directory where the calibration parameters will be saved.",
    )
    parser.set_defaults(
        func=lambda args: calibrate_camera(
            args.input,
            args.board_width,
            args.board_height,
            args.square_size,
            args.fisheye,
            args.output,
            (args.sensor_width, args.sensor_height),
        ),
    )
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calibrate Camera Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparser = register_subparser(subparsers)
    # Load defaults into the subparser if config is given
    args, _ = parser.parse_known_args()
    if args.config and args.command:
        load_yaml_defaults(subparser, args.config)
    args = parser.parse_args()
    args.func(args)
