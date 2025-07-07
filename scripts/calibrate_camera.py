from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np
import yaml
from tqdm import tqdm


def calibrate_camera(
    input_dir: str,
    board_width: int,
    board_height: int,
    square_size: float,
    fisheye: bool,
    output_file: str,
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

    if not fisheye:
        ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
            objpoints,
            imgpoints,
            gray.shape[::-1],
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

    calibration_data = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "reprojection_error": float(ret),
    }

    with open(output_file, "w") as f:
        yaml.dump(calibration_data, f)

    print(f"Camera calibration saved to: {output_file}")


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    """
    Registers the 'calibrate-camera' subparser for CLI.

    Args:
        subparsers (argparse._SubParsersAction): The subparsers object from the main parser.
    """
    parser = subparsers.add_parser(
        "calibrate-camera",
        help="Calibrate a camera using extracted chessboard images.",
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
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calibrate Camera Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_subparser(subparsers)
    args = parser.parse_args()
    args.func(args)
