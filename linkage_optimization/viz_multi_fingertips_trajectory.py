import pickle

import numpy as np
import plotly.graph_objects as go
import zarr

from dexumi.common.utility.matrix import (
    construct_coordinate,
    construct_coordinate_general,
    convert_homogeneous_matrix,
    invert_transformation,
    visualize_frames_and_points,
    visualize_multiple_frames_and_points,
    visualize_point_groups,
)

frame_mocap_diameter = 0.0095

# Load data
buffer = zarr.open(
    "hardware_design_data/inspire_mocap/inspire_hand_mocap_1-6-low_residual_0.39",
    mode="r",
)
viz_frames = {}
for finger_name in [
    "middle",
    "index",
    "ring",
    "little",
    "thumb_swing",
    "thumb_proximal_0",
    "thumb_proximal_100",
    "thumb_proximal_200",
    "thumb_proximal_300",
    "thumb_proximal_400",
    "thumb_proximal_600",
    "thumb_proximal_800",
    "thumb_proximal_1000",
]:
    episode = buffer[finger_name]["episode_0"]
    finger = episode["fingertips_marker"]
    frame = episode["inspire_hand_marker"]

    R_flange_frame = np.eye(3)
    t_flange_frame = np.array([0.0, -0.1008, -0.027 + frame_mocap_diameter / 2])
    T_flange_frame = convert_homogeneous_matrix(R_flange_frame, t_flange_frame)

    # Print shapes
    print(finger.shape, frame.shape)

    # Create a homogeneous transformation matrix for the frame
    # origin = (frame[:, 0, :] + frame[:, 1, :]) / 2
    # x_axis = frame[:, 0, :]
    # y_axis = frame[:, 2, :]

    # origin = (frame[:, 0, :] + frame[:, 2, :]) / 2
    # x_axis = frame[:, 2, :]
    # y_axis = frame[:, 1, :]

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
    # visualize_point_groups({"frame": frame, "finger": finger})
    # padding ones to finger
    finger = np.concatenate(
        [finger, np.ones((finger.shape[0], finger.shape[1], 1))], axis=2
    )

    # padding ones to frame
    frame = np.concatenate(
        [frame, np.ones((frame.shape[0], frame.shape[1], 1))], axis=2
    )

    # compute world_flange homogeneous transformation matrix
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
                cube_origin[i], cube_x_axis[i], cube_z_axis[i], axis1="-x", axis2="-z"
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
    # T_flange_cube = T_flange_world @ T_world_cube
    for T_world_cube, T_world_flange in zip(cube_homogenous_matrices, T_world_flanges):
        T_flange_center = invert_transformation(T_world_flange) @ T_world_cube
        T_flange_cubes.append(T_flange_center)
    T_flange_cubes = np.array(T_flange_cubes)
    T_flange_cubes[:, 1, -1] = T_flange_cubes[:, 1, -1] + 0.03070
    viz_frames[finger_name] = T_flange_cubes
    # print(T_flange_cubes_adjusted - T_flange_cubes)[:, 1, -1]
    # viz_frames[f"{finger_name}_adj_frame"] = T_flange_cubes_adjusted


visualize_multiple_frames_and_points(
    frames_dict=viz_frames,
    show_axes=True,
    axis_length=0.01,
    save_path=f"viz_finger.html",
)

# Save the visualization frames dictionary to a pickle file
with open("viz_1-6_low_residual_frames.pkl", "wb") as f:
    pickle.dump(viz_frames, f)
