import click
import numpy as np
import scipy.spatial.transform as st
import zarr
from scipy.optimize import minimize

from dexumi.common.utility.matrix import (
    homogeneous_matrix_to_6dof,
    invert_transformation,
    relative_transformation,
    vec6dof_to_homogeneous_matrix,
)


@click.command()
@click.option("-rp", "--record_path", type=str, default="iphone_calibration")
@click.option("--record_fps", type=int, default=45, help="record fps")
@click.option("--episode", type=int, default=0, help="episode index")
@click.option("--ur_latency", type=float, default=0.16, help="UR5 latency in seconds")
@click.option(
    "--iphone_latency", type=float, default=0.185, help="iPhone latency in seconds"
)
@click.option(
    "--optimize_rotation", is_flag=True, help="Optimize both rotation and translation"
)
def main(
    record_path, record_fps, episode, ur_latency, iphone_latency, optimize_rotation
):
    # Load initial data
    record_buffer = zarr.open(record_path, mode="r")
    episode = record_buffer[f"episode_{episode}"]
    iphone_pose = episode["iphone_pose"]
    ee_pose = np.array(
        [vec6dof_to_homogeneous_matrix(p[:3], p[3:]) for p in episode["ee_pose"]]
    )
    offset = np.round((iphone_latency - ur_latency) / (1 / record_fps)).astype(int)
    iphone_pose = iphone_pose[offset:]
    ee_pose = ee_pose[:-offset]
    initial_pose = ee_pose[0]

    relative_pose = np.array(
        [relative_transformation(iphone_pose[0], p) for p in iphone_pose]
    )

    T_ET = np.array(
        [
            [0, -1, 0, -0.0],
            [-1, 0, 0, -0.0],
            [0, 0, -1, 0.0],
            [0, 0, 0, 1],
        ]
    )

    def cost_function(x, relative_pose, ee_pose, initial_pose, optimize_rotation):
        """
        The objective function to minimize.
        x: Contains 3 or 6 parameters, 3 for translation and optionally 3 for rotation
        """
        if optimize_rotation:
            # Extract translation and rotation vector
            translation = x[:3]
            rotation_vector = x[3:6]
        else:
            # Extract translation only, use fixed rotation
            translation = x
            rotation_vector = st.Rotation.from_matrix(T_ET[:3, :3]).as_rotvec()

        # Create the transformation matrix T_ET
        T_ET_opt = vec6dof_to_homogeneous_matrix(translation, rotation_vector)

        # Compute the estimated transformation matrices
        T_BE = initial_pose

        # Calculate the new replay 6DOF positions using the optimized T_ET
        replay_6dof = []
        for rel_pose in relative_pose:
            T_BN_optimized = (
                T_BE @ T_ET_opt @ rel_pose @ invert_transformation(T_ET_opt)
            )
            replay_6dof.append(homogeneous_matrix_to_6dof(T_BN_optimized))

        replay_6dof = np.array(replay_6dof)
        actual_6dof = np.array([homogeneous_matrix_to_6dof(p) for p in ee_pose])

        # Calculate the difference between the actual and replayed positions
        error = np.linalg.norm(replay_6dof - actual_6dof, axis=1)
        return np.mean(error)

    # Initial rotation vector
    initial_rotation = st.Rotation.from_matrix(T_ET[:3, :3]).as_rotvec()

    if optimize_rotation:
        # Initial guess for the optimization (both translation and rotation)
        initial_guess = np.concatenate([T_ET[:3, 3], initial_rotation])

        # Optimization bounds
        bounds = [
            (-0.2, 0.2),
            (-0.2, 0.2),
            (-0.2, 0.2),  # Translation bounds
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),  # Rotation bounds
        ]
    else:
        # Initial guess for the optimization (translation only)
        initial_guess = T_ET[:3, 3]

        # Optimization bounds
        bounds = [
            (-0.2, 0.2),
            (-0.2, 0.2),
            (-0.2, 0.2),  # Translation bounds
        ]

    # Run the optimization
    result = minimize(
        cost_function,
        initial_guess,
        args=(relative_pose, ee_pose, initial_pose, optimize_rotation),
        bounds=bounds,
        method="L-BFGS-B",
    )

    if optimize_rotation:
        # Extract the optimized translation and rotation vector
        optimized_translation = result.x[:3]
        optimized_rotation = result.x[3:6]
    else:
        # Extract the optimized translation and use the initial rotation
        optimized_translation = result.x
        optimized_rotation = initial_rotation

    optimized_T_ET = vec6dof_to_homogeneous_matrix(
        optimized_translation, optimized_rotation
    )

    print(f"Optimization error: {result.fun}")
    print("Optimized T_ET:")
    print(optimized_T_ET)

    # Print the optimized values separately for clarity
    print("\nOptimized translation:", optimized_translation)
    print(
        "Optimized rotation matrix:",
        st.Rotation.from_rotvec(optimized_rotation).as_matrix(),
    )


if __name__ == "__main__":
    main()
