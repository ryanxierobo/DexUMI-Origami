import json
import os
import pickle
from functools import partial
from multiprocessing import Pool

import click
import numpy as np
import scipy.spatial.transform as st
import tqdm
import zarr
from inspire_urdf_writer import urdf_writer
from scipy.optimize import minimize

from dexumi.common.utility.matrix import (
    construct_coordinate,
    construct_coordinate_general,
    convert_homogeneous_matrix,
    homogeneous_matrix_to_6dof,
    homogeneous_matrix_to_xyzwxzy,
    invert_transformation,
    visualize_multiple_frames_and_points,
)

FINGRES = [
    "thumb_proximal_0",
    "thumb_proximal_300",
    # "thumb_proximal_500",
    "thumb_proximal_600",
    # "thumb_proximal_800",
    "thumb_proximal_1000",
]
BOUNDS = (
    [(-np.pi * 3 / 4, 0)] * len(FINGRES)  # bounds for each angle
    + [(-np.pi / 20, np.pi / 20)]  # bounds for rot around x
    + [(-0.1, 0.0)]  # bounds for x
    + [(-0.04, 0.018)]  # bounds for y
    + [(-0.030, -0.018)]  # bounds for z
)


def compute_total_loss(x, gt_fingertip_poses, T_fingerbase_sample):
    """
    Compute the total loss with individual angles and x positions but shared x,y and z.
    x: parameters [angle*len(FINGRES), shared_x, shared_y, shared_z]
    """
    total_loss = 0
    angles = x[: len(FINGRES)]  # Individual angles for each finger
    rot_x = x[len(FINGRES)]
    shared_xyz = x[len(FINGRES) + 1 :]  # Shared y and z parameters
    for i, (finger_name, gt_pose) in enumerate(gt_fingertip_poses.items()):
        rotation = st.Rotation.from_euler(
            "zxy", [np.pi / 2, np.pi / 2 + rot_x, angles[i]]
        ).as_matrix()
        position = np.array(
            [shared_xyz[0], shared_xyz[1], shared_xyz[2]]
        )  # Combine individual x with shared y,z
        T_flange_fingerbase = convert_homogeneous_matrix(R=rotation, p=position)

        transformed_pose = np.array(
            [
                homogeneous_matrix_to_xyzwxzy(T_flange_fingerbase @ s)
                for s in T_fingerbase_sample
            ]
        )
        gt_fingertip_pose = [homogeneous_matrix_to_xyzwxzy(p) for p in gt_pose]
        gt_fingertip_pose = np.array(gt_fingertip_pose)

        # Calculate exponential weights
        weights = np.exp(-0.0 * np.arange(len(gt_fingertip_pose)))
        # weights = gaussian_weights(len(gt_fingertip_pose))
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        for k, pose in enumerate(gt_fingertip_pose):
            weight = weights[k]
            t_dists = np.linalg.norm(transformed_pose[:, :3] - pose[:3], axis=1)
            r_dists = np.linalg.norm(transformed_pose[:, 3:] - pose[3:], axis=1)
            dists = t_dists + 0.005 * r_dists
            total_loss += weight * np.min(dists)
    return total_loss


def optimize_trajectory(args):
    """
    Optimize a trajectory with individual angles and x positions but shared y and z.
    """
    initial_guess, bounds, gt_fingertip_poses, T_fingerbase_sample = args

    result = minimize(
        compute_total_loss,
        initial_guess,
        args=(gt_fingertip_poses, T_fingerbase_sample),
        method="L-BFGS-B",
        bounds=bounds,
    )
    return result.x, result.fun


def process_share_design(args):
    """
    Process a single share design configuration.
    """
    share_design_key, data = args
    share_design_group, viz_frames, T_W1_W2, sample_trajectory_downsample_ratio = data

    # Process sample trajectories
    sub_sample_trajectories = []
    sub_sample_mapping = []
    for design_key in list(share_design_group.array_keys()):
        sample_trajectory = share_design_group[design_key]
        v = T_W1_W2 @ sample_trajectory[::sample_trajectory_downsample_ratio]
        sub_sample_trajectories.append(v)
        sub_sample_mapping.append(design_key)

    # Initial guess: len(FINGRES) angles + 1 shared x_rot+ 3 shared x,y,z parameters
    initial_guess = np.zeros(len(FINGRES) + 1 + 3)

    # Prepare finger data
    gt_fingertip_poses = {finger: viz_frames[finger] for finger in FINGRES}

    # Optimize for each trajectory
    best_result = None
    best_loss = float("inf")
    best_design = None

    for traj, design_key in zip(sub_sample_trajectories, sub_sample_mapping):
        opt_args = (initial_guess, BOUNDS, gt_fingertip_poses, traj)
        result, loss = optimize_trajectory(opt_args)

        if loss < best_loss:
            best_loss = loss
            best_result = result
            best_design = design_key

    # Format results for each finger
    finger_results = {}
    for i, finger in enumerate(FINGRES):
        finger_results[finger] = {
            "design": best_design,
            "result": np.array(
                [
                    best_result[i],  # individual angle
                    best_result[len(FINGRES)],  # shared_x_rot
                    best_result[len(FINGRES) + 1],  # shared x
                    best_result[len(FINGRES) + 2],  # shared y
                    best_result[len(FINGRES) + 3],  # shared z
                ]
            ),
            "loss": best_loss / len(FINGRES),  # approximate per-finger loss
        }

    return {
        "share_design_key": share_design_key,
        "total_loss": best_loss,
        "finger_results": finger_results,
    }


def get_n_jobs(percentage):
    total_cores = os.cpu_count()
    return max(1, int(total_cores * percentage / 100))


@click.command()
@click.option(
    "-r",
    "--res-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to sim zarr buffer",
)
@click.option(
    "-b",
    "--buffer-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to gt zarr buffer",
)
@click.option(
    "-o",
    "--output-path",
    default="optimization_results",
    help="Path to zarr buffer",
)
@click.option(
    "-d",
    "--frame-mocap-diameter",
    type=float,
    default=0.0095,  # M4 Markers
    show_default=True,
    help="Frame mocap diameter in meters",
)
@click.option(
    "-s",
    "--sample-trajectory-downsample-ratio",
    type=int,
    default=1,
    show_default=True,
    help="Downsample ratio for sample trajectories",
)
@click.option(
    "-p",
    "--percentage-of-cores",
    default=60,
    help="Percentage of CPU cores to use",
)
def main(
    res_path,
    buffer_path,
    output_path,
    frame_mocap_diameter,
    sample_trajectory_downsample_ratio,
    percentage_of_cores,
):
    """Process finger data from res.pkl and zarr buffer with parallel processing."""
    os.makedirs(output_path, exist_ok=True)
    # Extract timestamp from res_path
    timestamp = res_path.split("_")[-1]
    output_path = os.path.join(output_path, timestamp)
    os.makedirs(output_path, exist_ok=True)

    # Load mocap zarr buffer
    buffer = zarr.open(buffer_path, mode="r")
    viz_frames = {}
    finger_to_optimize = FINGRES

    # Process fingers
    for finger_name in finger_to_optimize:
        episode = buffer[finger_name]["episode_0"]
        finger = episode["fingertips_marker"]
        frame = episode["inspire_hand_marker"]

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
        cube_origin = (finger[:, 0, :3] + finger[:, 3, :3]) / 2
        cube_x_axis = finger[:, 3, :3]
        cube_z_axis = finger[:, 2, :3]
        cube_homogenous_matrices = [
            construct_coordinate_general(
                cube_origin[i], cube_x_axis[i], cube_z_axis[i], axis1="-x", axis2="-z"
            )
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
        # TODO: add explaination
        T_flange_cubes[:, 1, -1] = T_flange_cubes[:, 1, -1] + 0.03070
        viz_frames[finger_name] = T_flange_cubes

    # Setup for parallel processing
    res_buffer = zarr.open(res_path, mode="r")["sweep_data"]
    meta_data = zarr.open(res_path, mode="r")["meta_data"]
    R_W1_W2 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    t_W1_W2 = np.array([0.0, 0.0, 0.0])
    T_W1_W2 = convert_homogeneous_matrix(R_W1_W2, t_W1_W2)

    # Prepare all share designs to process
    all_share_design_keys = list(res_buffer.group_keys())
    process_args = [
        (
            share_design_key,
            (
                res_buffer[share_design_key],
                viz_frames,
                T_W1_W2,
                sample_trajectory_downsample_ratio,
            ),
        )
        for share_design_key in all_share_design_keys
    ]

    # Process all share designs in parallel
    n_jobs = get_n_jobs(percentage_of_cores)
    with Pool(processes=n_jobs) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap(process_share_design, process_args),
                total=len(process_args),
                desc="Processing share designs",
            )
        )

    # Find best design
    best_result = min(results, key=lambda x: x["total_loss"])
    best_share_design = best_result["share_design_key"]

    print(
        "Best configuration:",
        {
            finger: [results["design"], results["result"].tolist()]
            for finger, results in best_result["finger_results"].items()
        },
    )
    viz_points = {
        f"optimized_{finger}_base": results["result"][2:].reshape(1, 1, -1)
        for finger, results in best_result["finger_results"].items()
    }
    float_values = [float(value) for value in best_share_design.split("_")]
    (
        angle_B_X1_A,
        angle_X1_A_C,
        angle_A_X1_X2,
        angle_X2_C_D,
        link_X1_X2_y,
        link_A_C_y,
        link_X2_C_y,
        link_x1_A_y,
        link_x1_B_y,
    ) = float_values

    # Generate URDFs and visualizations for each finger
    for finger, results in best_result["finger_results"].items():
        best_T_flange_fingerbase = convert_homogeneous_matrix(
            R=st.Rotation.from_euler(
                "zxy",
                [np.pi / 2, np.pi / 2 + results["result"][1], results["result"][0]],
            ).as_matrix(),
            p=results["result"][2:],  # shared xyz
        )
        best_transformed_trajectory = np.array(
            [
                best_T_flange_fingerbase @ T_W1_W2 @ p
                for p in res_buffer[best_share_design][results["design"]][:]
            ]
        )
        viz_frames[f"optimized_{finger}"] = best_transformed_trajectory
        best_design = results["design"]
        link_C_D_y = float(best_design)
        urdf_writer(
            link_C_D_y,
            link_X1_X2_y,
            link_A_C_y,
            link_X2_C_y,
            link_x1_A_y,
            link_x1_B_y,
            angle_B_X1_A,
            angle_X1_A_C,
            angle_A_X1_X2,
            angle_X2_C_D,
            output_path=f"{output_path}/{finger}.urdf",
            type="thumb",
        )

    print(f"""
    angle_B_X1_A: {angle_B_X1_A}
    angle_X1_A_C: {angle_X1_A_C}
    angle_A_X1_X2: {angle_A_X1_X2}
    angle_X2_C_D: {angle_X2_C_D}
    link_X1_X2_y: {link_X1_X2_y}
    link_A_C_y: {link_A_C_y}
    link_X2_C_y: {link_X2_C_y}
    link_x1_A_y: {link_x1_A_y}
    link_x1_B_y: {link_x1_B_y}
    """)
    # Save results to jsonl
    result_dict = {
        "best_share_design_params": {
            "angle_B_X1_A": angle_B_X1_A,
            "angle_X1_A_C": angle_X1_A_C,
            "angle_A_X1_X2": angle_A_X1_X2,
            "angle_X2_C_D": angle_X2_C_D,
            "link_X1_X2_y": link_X1_X2_y,
            "link_A_C_y": link_A_C_y,
            "link_X2_C_y": link_X2_C_y,
            "link_x1_A_y": link_x1_A_y,
            "link_x1_B_y": link_x1_B_y,
        },
        "best_configuration": {
            finger: {
                "design": results["design"],
                "result": results["result"].tolist(),
            }
            for finger, results in best_result["finger_results"].items()
        },
        "bounds": BOUNDS,
    }
    total_loss = 0
    for finger, results in best_result["finger_results"].items():
        total_loss += results["loss"]
    result_dict["overall_loss"] = total_loss
    # Add sweep ranges to result dictionary
    result_dict["sweep_ranges"] = {
        "angle_B_X1_A_sweep": meta_data["angle_B_X1_A_sweep"][:].tolist(),
        "angle_X1_A_C_sweep": meta_data["angle_X1_A_C_sweep"][:].tolist(),
        "angle_A_X1_X2_sweep": meta_data["angle_A_X1_X2_sweep"][:].tolist(),
        "angle_X2_C_D_sweep": meta_data["angle_X2_C_D_sweep"][:].tolist(),
        "link_C_D_y_sweep": meta_data["link_C_D_y_sweep"][:].tolist(),
        "link_X1_X2_y_sweep": meta_data["link_X1_X2_y_sweep"][:].tolist(),
        "link_A_C_y_sweep": meta_data["link_A_C_y_sweep"][:].tolist(),
        "link_X2_C_y_sweep": meta_data["link_X2_C_y_sweep"][:].tolist(),
        "link_x1_A_y_sweep": meta_data["link_x1_A_y_sweep"][:].tolist(),
        "link_x1_B_y_sweep": meta_data["link_x1_B_y_sweep"][:].tolist(),
    }
    res_save_path = os.path.join(output_path, "optimization_results.jsonl")
    with open(res_save_path, "w") as f:
        for k, v in result_dict.items():
            f.write(json.dumps({k: v}) + "\n")

    print(meta_data["angle_B_X1_A_sweep"][:])
    print("\nPosition visualizations:")
    print(
        f"angle_B_X1_A: {create_progress_bar(angle_B_X1_A, meta_data['angle_B_X1_A_sweep'][:])}"
    )
    print(
        f"angle_X1_A_C: {create_progress_bar(angle_X1_A_C, meta_data['angle_X1_A_C_sweep'][:])}"
    )
    print(
        f"angle_A_X1_X2: {create_progress_bar(angle_A_X1_X2, meta_data['angle_A_X1_X2_sweep'][:])}"
    )
    print(
        f"angle_X2_C_D: {create_progress_bar(angle_X2_C_D, meta_data['angle_X2_C_D_sweep'][:])}"
    )
    print(
        f"link_C_D_y: {create_progress_bar(link_C_D_y, meta_data['link_C_D_y_sweep'][:])}"
    )
    print(
        f"link_X1_X2_y: {create_progress_bar(link_X1_X2_y, meta_data['link_X1_X2_y_sweep'][:])}"
    )
    print(
        f"link_A_C_y: {create_progress_bar(link_A_C_y, meta_data['link_A_C_y_sweep'][:])}"
    )
    print(
        f"link_X2_C_y: {create_progress_bar(link_X2_C_y, meta_data['link_X2_C_y_sweep'][:])}"
    )
    print(
        f"link_x1_A_y: {create_progress_bar(link_x1_A_y, meta_data['link_x1_A_y_sweep'][:])}"
    )
    print(
        f"link_x1_B_y: {create_progress_bar(link_x1_B_y, meta_data['link_x1_B_y_sweep'][:])}"
    )

    # Add this after the progress bar prints
    output_txt = os.path.join(output_path, "progress_bars.txt")
    with open(output_txt, "w") as f:
        f.write("\nPosition visualizations:\n")
        f.write(
            f"angle_B_X1_A: {create_progress_bar(angle_B_X1_A, meta_data['angle_B_X1_A_sweep'][:])}\n"
        )
        f.write(
            f"angle_X1_A_C: {create_progress_bar(angle_X1_A_C, meta_data['angle_X1_A_C_sweep'][:])}\n"
        )
        f.write(
            f"angle_A_X1_X2: {create_progress_bar(angle_A_X1_X2, meta_data['angle_A_X1_X2_sweep'][:])}\n"
        )
        f.write(
            f"angle_X2_C_D: {create_progress_bar(angle_X2_C_D, meta_data['angle_X2_C_D_sweep'][:])}\n"
        )
        f.write(
            f"link_C_D_y: {create_progress_bar(link_C_D_y, meta_data['link_C_D_y_sweep'][:])}\n"
        )
        f.write(
            f"link_X1_X2_y: {create_progress_bar(link_X1_X2_y, meta_data['link_X1_X2_y_sweep'][:])}\n"
        )
        f.write(
            f"link_A_C_y: {create_progress_bar(link_A_C_y, meta_data['link_A_C_y_sweep'][:])}\n"
        )
        f.write(
            f"link_X2_C_y: {create_progress_bar(link_X2_C_y, meta_data['link_X2_C_y_sweep'][:])}\n"
        )
        f.write(
            f"link_x1_A_y: {create_progress_bar(link_x1_A_y, meta_data['link_x1_A_y_sweep'][:])}\n"
        )
        f.write(
            f"link_x1_B_y: {create_progress_bar(link_x1_B_y, meta_data['link_x1_B_y_sweep'][:])}\n"
        )
    print(f"Progress bars saved to {output_txt}")

    visualize_multiple_frames_and_points(
        frames_dict=viz_frames,
        points_dict=viz_points,
        show_axes=True,
        axis_length=0.01,
        save_path=f"{output_path}/eq_finger.html",
    )
    print("Results saved to", output_path)


def create_progress_bar(value, sweep_array):
    """Create minimal progress bar showing only sweep points and current value.

    Args:
        value: Current value to show
        sweep_array: Array of sweep points
    """
    if len(sweep_array) == 1:
        return "*"

    # Create string of | markers
    bar = ["|"] * len(sweep_array)

    # Find closest sweep point to value
    closest_idx = np.abs(sweep_array - value).argmin()
    bar[closest_idx] = "*"

    return "".join(bar)


if __name__ == "__main__":
    main()
