from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from scripts.lib.utils import get_config_parser
from scripts.lib.utils import load_camera_parameters
from scripts.lib.utils import load_yaml_defaults

COMMAND_NAME = "undistort-image"


def undistort_image(
    image_path: str,
    intrinsics_path: str,
    output_path: str,
    fisheye: bool,
) -> None:
    """
    Undistorts an image using camera intrinsics.

    Args:
        image_path (str): Path to the distorted image.
        intrinsics_path (str): Path to the YAML file with camera_matrix and dist_coeffs.
        output_path (str): Path to save the undistorted image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not os.path.exists(intrinsics_path):
        raise FileNotFoundError(f"Intrinsics file not found: {intrinsics_path}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image: {image_path}")

    # Load calibration data
    calib = load_camera_parameters(intrinsics_path)
    camera_matrix = np.array(calib["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.array(calib["dist_coeffs"], dtype=np.float32)

    h, w = image.shape[:2]

    # Undistort
    if not fisheye:
        new_camera_matrix = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            dist_coeffs,
            (w, h),
            0,
        )[0]
        undistorted = cv2.undistort(
            image,
            camera_matrix,
            dist_coeffs,
            None,
            new_camera_matrix,
        )
    else:
        # Compute optimal new camera matrix for fisheye
        new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            camera_matrix,
            dist_coeffs,
            (w, h),
            np.eye(3),
            balance=0.0,
        )
        undistorted = cv2.fisheye.undistortImage(
            image,
            camera_matrix,
            dist_coeffs,
            Knew=new_camera_matrix,
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, undistorted)
    print(f"✅ Undistorted image saved to: {output_path}")


def register_subparser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """
    Registers the 'undistort-image' subcommand for CLI.

    Args:
        subparsers (argparse._SubParsersAction): The subparser object to register with.
    """
    parser = subparsers.add_parser(
        COMMAND_NAME,
        help="Undistort a photo using camera calibration intrinsics.",
        parents=[get_config_parser()],
        conflict_handler="resolve",
    )
    parser.add_argument(
        "--distorted-image",
        help="Path to the distorted input image.",
    )
    parser.add_argument(
        "--intrinsics",
        help="Path to the YAML file with camera_matrix and dist_coeffs.",
    )
    parser.add_argument(
        "--fisheye",
        action="store_true",
        help="Flag that determines the camera model used.",
    )
    parser.add_argument(
        "--undistorted-image",
        help="Path to save the undistorted output image.",
    )

    def run(args):
        undistort_image(
            args.distorted_image,
            args.intrinsics,
            args.undistorted_image,
            args.fisheye,
        )

    parser.set_defaults(func=run)
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Undistortion Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparser = register_subparser(subparsers)
    # Load defaults into the subparser if config is given
    args, _ = parser.parse_known_args()
    if args.config and args.command:
        load_yaml_defaults(subparser, args.config)
    args = parser.parse_args()
    args.func(args)
