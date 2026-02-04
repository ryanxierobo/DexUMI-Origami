import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

import click
import numpy as np
import placo
import zarr
from inspire_urdf_writer import urdf_writer
from tqdm import tqdm

# Set environment variables for better performance
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def is_monotonic(arr):
    """
    Check if a list of float numbers is monotonic (either increasing or decreasing).
    Returns True if the list is monotonic, False otherwise.

    Args:
        arr (list): List of float numbers

    Returns:
        bool: True if list is monotonic, False otherwise
    """
    if len(arr) <= 1:
        return True

    # Check if monotonic increasing
    increasing = all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))
    # Check if monotonic decreasing
    decreasing = all(arr[i] >= arr[i + 1] for i in range(len(arr) - 1))

    return increasing or decreasing


def run_sim(urdf_path):
    robot = placo.RobotWrapper(
        urdf_path,
        placo.Flags.ignore_collisions,
    )
    solver = placo.KinematicsSolver(robot)
    solver.mask_fbase(True)
    solver.enable_velocity_limits(True)

    closing_task = solver.add_relative_position_task(
        "closing_link_1", "closing_link_2", np.zeros(3)
    )
    closing_task.configure("closing", "hard", 1.0)
    closing_task.mask.set_axises("yz")

    joints_task = solver.add_joints_task()
    joints_task.set_joints({"joint_X1_X2": 0})
    joints_task.configure("joint_X1_X2", "soft", 1.0)
    dt = 0.02
    t_arr = np.arange(0, np.pi / 2, dt)
    fingertip_trajectory = []
    solver.dt = dt

    # joint_task_errors = []
    for t in t_arr:
        joints_task.set_joints({"joint_X1_X2": t})
        solver.solve(True)
        robot.update_kinematics()
        # joint_task_errors.append(joints_task.error)
        T_finger_world = robot.get_T_world_frame("new_fingertips_link")
        fingertip_trajectory.append(T_finger_world)
    # Calculate distances from initial point using vectorized operations
    init_point = fingertip_trajectory[0][:3, 3]
    points = np.array([T[:3, 3] for T in fingertip_trajectory])
    distances = np.linalg.norm(points - init_point, axis=1)

    # Check if distances are monotonic
    if not is_monotonic(distances):
        raise ValueError("Fingertip trajectory is not monotonic")

    return np.array(fingertip_trajectory)


def batch_run_simulations(param_batch, worker_id, type):
    """Run multiple simulations in a single process"""
    results = []
    temp_dir = f"/dev/shm/linkage_sim/worker_{worker_id}"
    os.makedirs(temp_dir, exist_ok=True)
    print("len(param_batch): ", len(param_batch))
    assert type in ["finger", "thumb"]
    with tqdm(
        enumerate(param_batch),
        total=len(param_batch),
        desc=f"Worker {worker_id}",
        position=worker_id,
    ) as pbar:
        for i, params in pbar:
            try:
                (
                    angle_B_X1_A,
                    angle_X1_A_C,
                    angle_A_X1_X2,
                    angle_X2_C_D,
                    link_C_D_y,
                    link_X1_X2_y,
                    link_A_C_y,
                    link_X2_C_y,
                    link_x1_A_y,
                    link_x1_B_y,
                ) = params
                output_path = f"{temp_dir}/sim_{i}.urdf"

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
                    output_path=output_path,
                    type=type,
                )

                fingertip_trajectory = run_sim(urdf_path=output_path)
                key = f"{angle_B_X1_A}_{angle_X1_A_C}_{angle_A_X1_X2}_{angle_X2_C_D}_{link_C_D_y}_{link_X1_X2_y}_{link_A_C_y}_{link_X2_C_y}_{link_x1_A_y}_{link_x1_B_y}"
                results.append((key, fingertip_trajectory))

                # Clean up URDF file after use
                os.remove(output_path)

            except Exception as e:
                # print(f"Error in simulation {i} of worker {worker_id}: {str(e)}")
                continue

    return results


@click.command()
@click.option(
    "--type", default="finger", help="Type of simulation to run (finger or thumb)"
)
@click.option(
    "-p",
    "--percentage-of-cores",
    default=60,
    help="Percentage of CPU cores to use",
)
@click.option(
    "-s",
    "--save_path",
    default="/store/real/mengda/DexUMI/data_local",
    help="Sweep save path",
)
def main(type, percentage_of_cores, save_path):
    # Define parameter sweeps based on type
    # Optimize worker and batch configuration
    N_CORES = os.cpu_count()
    N_WORKERS = max(
        1, int(N_CORES * percentage_of_cores / 100)
    )  # Use specified percentage of cores
    print("====================================")
    print("sweeping", type)
    print("====================================")

    if type == "thumb":
        print("Running thumb simulation")
        link_x1_A_y_sweep = np.array([0.08, 0.010, 0.0125, 0.016, 0.0185])
        link_x1_B_y_sweep = np.array([0.015, 0.0175, 0.0195, 0.0215, 0.0235])
        angle_B_X1_A_sweep = np.array(
            [
                # np.pi * 20 / 180,
                # np.pi * 40 / 180,
                # np.pi * 50 / 180,
                np.pi * 60 / 180,
            ]
        )

        angle_X1_A_C_sweep = np.arange(
            np.pi / 4 + np.pi / 8, np.pi / 4 + np.pi / 4, np.pi / 8
        )
        angle_A_X1_X2_sweep = np.arange(
            np.pi / 2,
            np.pi / 2 + np.pi / 4,
            np.pi / 8,
        )
        angle_X2_C_D_sweep = np.arange(
            np.pi / 2,
            np.pi / 2 + np.pi / 4,
            np.pi / 8,
        )
        link_C_D_y_sweep = np.array([0.038, 0.040, 0.042, 0.044, 0.046, 0.048])
        link_X2_C_y_sweep = np.arange(0.008, 0.036, 0.004)
        link_A_C_y_sweep = np.arange(0.04, 0.16, 0.02)
        link_X1_X2_y_sweep = np.arange(0.04, 0.16, 0.02)
    else:
        print("Running finger simulation")
        angle_B_X1_A_sweep = np.array([50 / 180 * np.pi])
        angle_X1_A_C_sweep = np.arange(
            np.pi / 4 - np.pi / 16, np.pi / 4 + np.pi / 8, np.pi / 48
        )
        angle_A_X1_X2_sweep = np.arange(
            np.pi / 2 + np.pi / 8 - np.pi / 4,
            np.pi / 2 + np.pi / 8 + np.pi / 4,
            np.pi / 48,
        )
        angle_X2_C_D_sweep = np.arange(
            np.pi / 2 + np.pi / 8, np.pi / 2 + np.pi / 4 + np.pi / 8, np.pi / 48
        )
        link_C_D_y_sweep = np.arange(0.036, 0.056, 0.001)
        link_X1_X2_y_sweep = np.array([0.032])
        link_A_C_y_sweep = np.array([0.030, 0.031, 0.032, 0.033, 0.034, 0.035, 0.036])
        # link_x1_A_y_sweep = np.array([0.007, 0.009, 0.011])
        link_x1_A_y_sweep = np.array([0.007])
        link_X2_C_y_sweep = np.arange(0.006, 0.012, 0.001)
        link_x1_B_y_sweep = np.array([0.0185])
    print("Parameter sweep sizes:")
    print(f"angle_B_X1_A_sweep: {len(angle_B_X1_A_sweep)}")
    print(f"angle_X1_A_C_sweep: {len(angle_X1_A_C_sweep)}")
    print(f"angle_A_X1_X2_sweep: {len(angle_A_X1_X2_sweep)}")
    print(f"angle_X2_C_D_sweep: {len(angle_X2_C_D_sweep)}")
    print(f"link_C_D_y_sweep: {len(link_C_D_y_sweep)}")
    print(f"link_X1_X2_y_sweep: {len(link_X1_X2_y_sweep)}")
    print(f"link_A_C_y_sweep: {len(link_A_C_y_sweep)}")
    print(f"link_X2_C_y_sweep: {len(link_X2_C_y_sweep)}")
    print(f"link_x1_A_y_sweep: {len(link_x1_A_y_sweep)}")
    print(f"link_x1_B_y_sweep: {len(link_x1_B_y_sweep)}")

    # Generate all parameter combinations
    param_combinations = list(
        product(
            angle_B_X1_A_sweep,
            angle_X1_A_C_sweep,
            angle_A_X1_X2_sweep,
            angle_X2_C_D_sweep,
            link_C_D_y_sweep,
            link_X1_X2_y_sweep,
            link_A_C_y_sweep,
            link_X2_C_y_sweep,
            link_x1_A_y_sweep,
            link_x1_B_y_sweep,
        )
    )

    # Calculate optimal batch size based on total combinations and available workers
    total_combinations = len(param_combinations)
    min_batches_per_worker = (
        4  # Ensure each worker gets at least 4 batches for better load balancing
    )
    batch_size = max(1, total_combinations // (N_WORKERS * min_batches_per_worker))

    # Split combinations into batches
    batches = [
        param_combinations[i : i + batch_size]
        for i in range(0, total_combinations, batch_size)
    ]

    print("total combinations: ", total_combinations)
    print("sample per worker: ", total_combinations // N_WORKERS)

    # Create temp directory in RAM
    os.makedirs("/dev/shm/linkage_sim", exist_ok=True)

    # Initialize results dictionary
    os.makedirs(save_path, exist_ok=True)
    save_path = f"{save_path}/sweep_{time.strftime('%Y%m%d%H%M')}"
    sweep_buffer = zarr.open(
        save_path,
        mode="w",
    )
    # save sweep parameters as meta_data
    meta_data = sweep_buffer.require_group("meta_data")
    meta_data["angle_B_X1_A_sweep"] = angle_B_X1_A_sweep
    meta_data["angle_X1_A_C_sweep"] = angle_X1_A_C_sweep
    meta_data["angle_A_X1_X2_sweep"] = angle_A_X1_X2_sweep
    meta_data["angle_X2_C_D_sweep"] = angle_X2_C_D_sweep
    meta_data["link_C_D_y_sweep"] = link_C_D_y_sweep
    meta_data["link_X1_X2_y_sweep"] = link_X1_X2_y_sweep
    meta_data["link_A_C_y_sweep"] = link_A_C_y_sweep
    meta_data["link_X2_C_y_sweep"] = link_X2_C_y_sweep
    meta_data["link_x1_A_y_sweep"] = link_x1_A_y_sweep
    meta_data["link_x1_B_y_sweep"] = link_x1_B_y_sweep

    sweep_data = sweep_buffer.require_group("sweep_data")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        # Submit batch tasks with worker IDs
        futures = [
            executor.submit(batch_run_simulations, batch, i % N_WORKERS, type)
            for i, batch in enumerate(batches)
        ]

        # Process results with progress bar
        valid_designs = 0
        for future in as_completed(futures):
            try:
                batch_results = future.result()
                for key, fingertip_trajectory in batch_results:
                    float_values = [float(value) for value in key.split("_")]
                    (
                        angle_B_X1_A,
                        angle_X1_A_C,
                        angle_A_X1_X2,
                        angle_X2_C_D,
                        link_C_D_y,
                        link_X1_X2_y,
                        link_A_C_y,
                        link_X2_C_y,
                        link_x1_A_y,
                        link_x1_B_y,
                    ) = float_values
                    share_design = sweep_data.require_group(
                        f"{angle_B_X1_A}_{angle_X1_A_C}_{angle_A_X1_X2}_{angle_X2_C_D}_{link_X1_X2_y}_{link_A_C_y}_{link_X2_C_y}_{link_x1_A_y}_{link_x1_B_y}"
                    )
                    share_design[f"{link_C_D_y}"] = fingertip_trajectory
                    valid_designs += 1
            except Exception as e:
                print(f"Error processing batch result: {str(e)}")
                print(f"Traceback:\n{traceback.format_exc()}")
                continue

    print(
        f"Completed {valid_designs} successful simulations among {total_combinations} combinations. Results saved to {save_path}"
    )


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
