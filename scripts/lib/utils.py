from __future__ import annotations

import argparse

import numpy as np
import yaml


def load_camera_parameters(yaml_path: str):
    """Loads camera intrinsics and distortion coefficients from a custom YAML file.

    Args:
        yaml_path (str): Path to the YAML file.

    Returns:
        dict: Dictionary with camera matrix and distortion coefficients (OpenCV compatible).
    """
    with open(yaml_path) as file:
        data = yaml.safe_load(file)

    # Extract intrinsics (in pixels)
    fx = data["focal_length_pixels"]["fx"]
    fy = data["focal_length_pixels"]["fy"]
    cx = data["principal_point_pixels"]["cx"]
    cy = data["principal_point_pixels"]["cy"]
    skew = data.get("skew", 0.0)

    camera_matrix = np.array(
        [
            [fx, skew, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # Extract distortion coefficients (OpenCV expects a flat array)
    radial = data["distortion_coefficients"]["radial"]
    tangential = data["distortion_coefficients"]["tangential"]

    dist_coeffs = np.array(
        [
            radial["k1"],
            radial["k2"],
            tangential["p1"],
            tangential["p2"],
            radial["k3"],
            radial.get("k4", 0.0),
            radial.get("k5", 0.0),
            radial.get("k6", 0.0),
        ],
        dtype=np.float64,
    )

    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
    }


def load_yaml_defaults(parser: argparse.ArgumentParser, config_file: str):
    if config_file:
        with open(config_file) as f:
            config = yaml.safe_load(f)

        # Filter config keys to match known argparse arguments
        valid_keys = {action.dest for action in parser._actions}
        filtered_config = {k: v for k, v in config.items() if k in valid_keys}
        parser.set_defaults(**filtered_config)


def get_config_parser():
    parser = argparse.ArgumentParser(
        "Config YAML File Parser",
        conflict_handler="resolve",
    )
    parser.add_argument("--config", type=str)
    return parser
