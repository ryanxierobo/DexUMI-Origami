import json
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET

import numpy as np
import placo
import scipy.spatial.transform as st
from ischedule import run_loop, schedule
from placo_utils.visualization import robot_frame_viz, robot_viz

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
        size="0.01 0.01 0.01",
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
    print(thumb_parm[1:])

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


if __name__ == "__main__":
    output_urdf_path = "opt_full.urdf"  # Replace with your output path
    modify_urdf(
        finger_optimization_path="hardware_design_data/optimization_results/2025_1_11/finger",
        # thumb_optimization_path="/home/mengda/Dev/exoskeleton/hardware_design/optimization_results/real_perfect_thumb",
        thumb_optimization_path="hardware_design_data/optimization_results/2025_1_11/thumb",
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
    thumb_swing_task.set_joints({"palm_to_thumb_proximal_0": 0})

    viz = robot_viz(robot)
    t = 0
    dt = 0.01
    solver.dt = dt
    last_targets = []
    last_target_t = 0

    # Define the lower and upper bounds for the regular fingers
    finger_lower_bound = 0
    finger_upper_bound = np.pi / 2

    # Define separate bounds for thumb proximal
    thumb_lower_bound = 0  # Adjust these values as needed
    thumb_upper_bound = np.pi / 2  # Adjust these values as needed

    # Calculate the amplitude and offset for regular fingers
    finger_amplitude = (finger_upper_bound - finger_lower_bound) / 2
    finger_offset = (finger_upper_bound + finger_lower_bound) / 2

    # Calculate the amplitude and offset for thumb
    thumb_amplitude = (thumb_upper_bound - thumb_lower_bound) / 2
    thumb_offset = (thumb_upper_bound + thumb_lower_bound) / 2

    @schedule(interval=dt)
    def loop():
        global t, last_targets, last_target_t
        t += dt

        # Moving regular fingers
        for finger_name, joints_task in zip(
            ["index", "middle", "ring", "little"],
            joints_tasks[:-1],  # Exclude thumb
        ):
            joint_value = finger_amplitude * np.sin(t) + finger_offset
            joints_task.set_joints({f"{finger_name}_joint_AC": joint_value})

        # Moving thumb with its own bounds
        thumb_joint_value = thumb_amplitude * np.sin(t) + thumb_offset
        joints_tasks[-1].set_joints({"thumb_proximal_0_joint_AC": thumb_joint_value})

        # Thumb swing remains unchanged
        thumb_swing_task.set_joints({"palm_to_thumb_proximal_0": np.pi / 2})

        # Solving the IK
        solver.solve(True)
        robot.update_kinematics()
        solver.dump_status()
        for finger_name in ["index", "middle", "ring", "little", "thumb_proximal_0"]:
            robot_frame_viz(robot, f"{finger_name}_new_fingertips_link")
        # Displaying the robot and the effector frame
        viz.display(robot.state.q)

    run_loop()
