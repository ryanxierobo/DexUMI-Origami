import json
import pickle
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
from collections import defaultdict

import click
import numpy as np
import placo
import scipy.spatial.transform as st
from ischedule import run_loop, schedule
from placo_utils.visualization import robot_viz

from dexumi.common.utility.matrix import visualize_multiple_frames_and_points
from dexumi.common.utility.urdf import (
    create_joint,
    create_link,
    update_link_and_joint_names_with_prefix,
)


def read_jsonl_to_dict(file_path):
    result = {}
    with open(file_path, "r") as file:
        for line in file:
            # Parse each line as JSON
            data = json.loads(line.strip())
            # Update the result dictionary with the parsed data
            result.update(data)
    return result


def modify_urdf(finger_optimization_path, thumb_optimization_path, output_path):
    # Load the optimization path
    finger_optimization_data = read_jsonl_to_dict(
        f"{finger_optimization_path}/optimization_results.jsonl"
    )
    thumb_optimization_data = read_jsonl_to_dict(
        f"{thumb_optimization_path}/optimization_results.jsonl"
    )

    # Create complete URDF for the optimization
    robot = ET.Element("robot", name="robot")
    # Create palm link
    create_link(
        robot=robot,
        name="link_palm",
        size="0.00001 0.00001 0.00001",
        xyz="0 0 0",
        rpy="0 0 0",
        rgba="1.0 1.0 1.0 1.0",
    )
    for finger_name, finger_design in finger_optimization_data[
        "best_configuration"
    ].items():
        finger_urdf_root = ET.parse(
            f"{finger_optimization_path}/{finger_name}.urdf"
        ).getroot()
        finger_parm = finger_design["result"]
        # modify_link_parameter(finger_urdf_root, "link_base", "rpy", f"0 0 {np.pi}")
        link_name_mapping = update_link_and_joint_names_with_prefix(
            finger_urdf_root, finger_name + "_"
        )
        for child in finger_urdf_root:
            robot.append(child)

        # Create joint connecting palm to finger base
        frame_rotation = st.Rotation.from_euler("xyz", [0, 0, np.pi], degrees=False)
        rotation = st.Rotation.from_euler("zxy", [0, 0, finger_parm[0]], degrees=False)
        urdf_rotation = (rotation * frame_rotation).as_euler("xyz", degrees=False)
        create_joint(
            robot=robot,
            name=f"palm_to_{finger_name}",
            joint_type="fixed",
            parent_link="link_palm",
            child_link=link_name_mapping["link_base"],  # Use mapped name of base link
            origin_xyz=f"{finger_parm[1]} {finger_parm[2]} {finger_parm[3]}",  # Adjust these values as needed
            origin_rpy=f"{urdf_rotation[0]} {urdf_rotation[1]} {urdf_rotation[2]}",  # Adjust these values as needed
            axis_xyz="1 0 0",
        )

    # # now load thumb
    thumb_name = "thumb_proximal_0"
    thumb_design = thumb_optimization_data["best_configuration"][thumb_name]
    thumb_urdf_root = ET.parse(f"{thumb_optimization_path}/{thumb_name}.urdf").getroot()
    # modify_joint_parameter(thumb_urdf_root, "joint_base_x1", "rpy", f"{np.pi} 0 0")
    thumb_parm = thumb_design["result"]
    link_name_mapping = update_link_and_joint_names_with_prefix(
        thumb_urdf_root, thumb_name + "_"
    )
    for child in thumb_urdf_root:
        robot.append(child)
    frame_rotation = st.Rotation.from_euler("xyz", [0, 0, np.pi], degrees=False)
    rotation = st.Rotation.from_euler(
        "zxy", [np.pi / 2, np.pi / 2 + thumb_parm[1], 0], degrees=False
    )
    urdf_rotation = (rotation * frame_rotation).as_euler("xyz", degrees=False)
    print(urdf_rotation / np.pi)
    # Create joint connecting palm to thumb base

    create_joint(
        robot=robot,
        name=f"palm_to_{thumb_name}",
        joint_type="revolute",
        parent_link="link_palm",
        child_link=link_name_mapping["link_base"],  # Use mapped name of base link
        origin_xyz=f"{thumb_parm[2]} {thumb_parm[3]} {thumb_parm[4]}",  # Adjust these values as needed
        origin_rpy=f"{urdf_rotation[0]} {urdf_rotation[1]} {urdf_rotation[2]}",  # Adjust these values as needed
        axis_xyz="0 0 1",
    )

    # Convert to pretty XML and save
    xml_str = ET.tostring(robot, encoding="unicode")
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)


@click.command()
@click.option(
    "--swing-move", is_flag=True, default=False, help="Enable thumb swing movement"
)
@click.option(
    "--finger-optimization-path",
    default="hardware_design_data/optimization_results/2025_1_11/finger",
    help="Path to finger optimization results",
    type=click.Path(exists=True),
)
@click.option(
    "--thumb-optimization-path",
    default="hardware_design_data/optimization_results/2025_1_11/thumb",
    help="Path to thumb optimization results",
    type=click.Path(exists=True),
)
def main(swing_move, finger_optimization_path, thumb_optimization_path):
    """Generate fingertip trajectories with optional thumb swing movement."""
    output_urdf_path = "opt_full.urdf"
    modify_urdf(
        finger_optimization_path=finger_optimization_path,
        thumb_optimization_path=thumb_optimization_path,
        output_path=output_urdf_path,
    )

    robot = placo.RobotWrapper(
        output_urdf_path,
        placo.Flags.ignore_collisions,
    )
    solver = placo.KinematicsSolver(robot)
    solver.mask_fbase(True)
    solver.enable_velocity_limits(True)
    joints_tasks = []
    for finger_name in ["index", "middle", "ring", "little", "thumb_proximal_0"]:
        # # Adding loop closing task
        closing_task = solver.add_relative_position_task(
            f"{finger_name}_closing_link_1",
            f"{finger_name}_closing_link_2",
            np.zeros(3),
        )
        closing_task.configure("closing", "hard", 1.0)
        closing_task.mask.set_axises("yz")

        # # Adding a task for the joints
        joints_task = solver.add_joints_task()
        # joints_task.set_joints({f"{finger_name}_joint_X1_X2": 0})
        # joints_task.configure(f"{finger_name}_joint_X1_X2", "soft", 1.0)
        joints_task.set_joints({f"{finger_name}_joint_AC": 0})
        joints_task.configure(f"{finger_name}_joint_AC", "soft", 1.0)
        joints_tasks.append(joints_task)

    thumb_swing_task = solver.add_joints_task()
    thumb_optimization_data = read_jsonl_to_dict(
        f"{thumb_optimization_path}/optimization_results.jsonl"
    )
    y_rot = thumb_optimization_data["best_configuration"]["thumb_proximal_0"]["result"][
        0
    ]
    y_rot = 2 * np.pi - y_rot
    if swing_move:
        thumb_swing_task.set_joints({"palm_to_thumb_proximal_0": 0})
    else:
        thumb_swing_task.set_joints({"palm_to_thumb_proximal_0": y_rot})
        print("thumb swing", y_rot)
    t = 0
    dt = 0.01
    solver.dt = dt
    t_arr = np.arange(0, np.pi / 2 + np.pi / 4, dt)
    for i in range(100):
        solver.solve(True)
        robot.update_kinematics()

    # Create defaultdict to store trajectories and joint values
    finger_trajectories = defaultdict(dict)
    # Initialize lists for each finger
    for finger in ["index", "middle", "ring", "little", "thumb_proximal_0"]:
        finger_trajectories[finger]["trajectory"] = []
        finger_trajectories[finger]["joint_values"] = []
    if swing_move:
        finger_trajectories["thumb_swing"]["trajectory"] = []
        finger_trajectories["thumb_swing"]["joint_values"] = []
    print("swing_move", swing_move)
    # while True:
    # viz = robot_viz(robot)
    for t in t_arr:
        joint_value = t

        for finger_name, joints_task in zip(
            ["index", "middle", "ring", "little", "thumb_proximal_0"], joints_tasks
        ):
            if swing_move and finger_name == "thumb_proximal_0":
                continue
            joints_task.set_joints({f"{finger_name}_joint_AC": joint_value})
        if swing_move:
            thumb_swing_task.set_joints({"palm_to_thumb_proximal_0": joint_value})
        else:
            thumb_swing_task.set_joints({"palm_to_thumb_proximal_0": y_rot})

        solver.solve(True)
        robot.update_kinematics()

        # Get transforms for each finger
        for finger in [
            "index",
            "middle",
            "ring",
            "little",
            "thumb_proximal_0",
        ]:
            T_finger_world = robot.get_T_world_frame(f"{finger}_new_fingertips_link")
            finger_trajectories[finger]["trajectory"].append(T_finger_world)
            finger_trajectories[finger]["joint_values"].append(joint_value)
            if swing_move and finger == "thumb_proximal_0":
                finger_trajectories["thumb_swing"]["trajectory"].append(T_finger_world)
                finger_trajectories["thumb_swing"]["joint_values"].append(joint_value)

    for k, v in finger_trajectories.items():
        for m, n in v.items():
            finger_trajectories[k][m] = np.array(n)
    visualize_multiple_frames_and_points(
        frames_dict={k: v["trajectory"] for k, v in finger_trajectories.items()},
        show_axes=True,
        axis_length=0.01,
    )
    # Save trajectories to pickle file
    with open("sim_fingertip_trajectory.pkl", "wb") as f:
        pickle.dump(finger_trajectories, f)


if __name__ == "__main__":
    main()
