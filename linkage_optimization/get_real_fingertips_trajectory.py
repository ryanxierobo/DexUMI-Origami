import json
import os
import pickle
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

import click
import numpy as np
import plotly.graph_objects as go
import scipy.spatial.transform as st
import tqdm
import zarr
from dexumi.common.utility.matrix import (
    construct_coordinate,
    construct_coordinate_general,
    convert_homogeneous_matrix,
    homogeneous_matrix_to_6dof,
    homogeneous_matrix_to_xyzwxzy,
    invert_transformation,
    visualize_multiple_frames_and_points,
)
from inspire_urdf_writer import urdf_writer
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

FINGERS = [
    "index",
    "middle",
    "ring",
    "little",
    "thumb_proximal_0",
    "thumb_proximal_400",
    "thumb_swing",
]


@click.command()
@click.option(
    "-b",
    "--buffer-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to zarr buffer",
)
@click.option(
    "-d",
    "--frame-mocap-diameter",
    type=float,
    default=0.0095,
    show_default=True,
    help="Frame mocap diameter in meters",
)
@click.option(
    "-y",
    "--y-offset",
    type=float,
    default=0.03070,
    show_default=True,
    help="Y offset in meters",
)
def main(
    buffer_path,
    frame_mocap_diameter,
    y_offset,
):
    # First analysis: Motor value prediction
    dex_buffer = zarr.open(buffer_path, mode="r")
    data = defaultdict(dict)
    for finger_name in FINGERS:
        episode = dex_buffer[finger_name]["episode_0"]
        finger = episode["fingertips_marker"]
        frame = episode["inspire_hand_marker"]
        motor_value = episode["motor_value"][:]

        R_flange_frame = np.eye(3)
        t_flange_frame = np.array([0.0, -0.1008, -0.027 + frame_mocap_diameter / 2])
        T_flange_frame = convert_homogeneous_matrix(R_flange_frame, t_flange_frame)

        origin = (frame[:, 0, :] + frame[:, 2, :]) / 2
        x_axis = frame[:, 0, :]
        y_axis = frame[:, 1, :]
        frame_homogenous_matrices = [
            construct_coordinate_general(
                origin[i], x_axis[i], y_axis[i], axis1="x", axis2="-y"
            )
            for i in range(len(origin))
        ]
        frame_homogenous_matrices = np.array(frame_homogenous_matrices)
        finger = np.concatenate(
            [finger, np.ones((finger.shape[0], finger.shape[1], 1))], axis=2
        )
        frame = np.concatenate(
            [frame, np.ones((frame.shape[0], frame.shape[1], 1))], axis=2
        )

        T_world_flanges = []
        for T_world_frame in frame_homogenous_matrices:
            T_world_flange = T_world_frame @ invert_transformation(T_flange_frame)
            T_world_flanges.append(T_world_flange)
        T_world_flanges = np.array(T_world_flanges)
        # Create a homogeneous transformation matrix for the cube_frame
        if "thumb" in finger_name:
            cube_origin = (finger[:, 0, :3] + finger[:, 3, :3]) / 2
            cube_x_axis = finger[:, 3, :3]
            cube_z_axis = finger[:, 2, :3]
            cube_homogenous_matrices = [
                construct_coordinate_general(
                    cube_origin[i],
                    cube_x_axis[i],
                    cube_z_axis[i],
                    axis1="-x",
                    axis2="-z",
                )
                for i in range(len(cube_origin))
            ]
        else:
            cube_origin = (finger[:, 1, :3] + finger[:, 3, :3]) / 2
            cube_x_axis = finger[:, 3, :3]
            cube_y_axis = finger[:, 0, :3]
            cube_homogenous_matrices = [
                construct_coordinate(cube_origin[i], cube_x_axis[i], cube_y_axis[i])
                for i in range(len(cube_origin))
            ]
        cube_homogenous_matrices = np.array(cube_homogenous_matrices)

        T_flange_cubes = []
        for T_world_cube, T_world_flange in zip(
            cube_homogenous_matrices, T_world_flanges
        ):
            T_flange_center = invert_transformation(T_world_flange) @ T_world_cube
            T_flange_cubes.append(T_flange_center)
        T_flange_cubes = np.array(T_flange_cubes)
        T_flange_cubes[:, 1, -1] = T_flange_cubes[:, 1, -1] + y_offset
        # Create a dictionary to store the data
        data[finger_name] = {
            "motor_value": motor_value[:-1],
            "T_flange_cubes": T_flange_cubes[1:],
        }

    # Save the data as a pickle file
    with open("real_fingertip_trajectory.pkl", "wb") as f:
        pickle.dump(data, f)
    visualize_multiple_frames_and_points(
        frames_dict={k: v["T_flange_cubes"] for k, v in data.items()},
        show_axes=True,
        axis_length=0.01,
    )


if __name__ == "__main__":
    main()
