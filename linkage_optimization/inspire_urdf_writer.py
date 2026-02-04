import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET

import numpy as np
from dexumi.common.utility.urdf import create_joint, create_link


def distance_point_to_line(A: np.array, C: np.array, X1: np.array) -> float:
    # Calculate the vector from A to C and from A to X1
    AC = C - A
    AX1 = X1 - A

    # Calculate the cross product of the vectors AC and AX1 (in 2D, this is the determinant)
    cross_product = np.abs(AC[0] * AX1[1] - AC[1] * AX1[0])

    # Calculate the length of the line segment AC (this is the denominator of the distance formula)
    line_length = np.linalg.norm(AC)

    # Distance from X1 to the line
    distance = cross_product / line_length
    return distance


def angle_between_lines(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
    # Calculate vectors AB and AC
    AB = B - A
    AC = C - A

    # Calculate dot product of AB and AC
    dot_product = np.dot(AB, AC)

    # Calculate magnitudes of AB and AC
    magnitude_AB = np.linalg.norm(AB)
    magnitude_AC = np.linalg.norm(AC)

    # Calculate the cosine of the angle
    cos_theta = dot_product / (magnitude_AB * magnitude_AC)

    # Ensure cos_theta is within the valid range for arccos due to floating point errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Calculate the angle in radians and convert to degrees
    angle_rad = np.arccos(cos_theta)
    # angle_deg = np.degrees(angle_rad)

    return angle_rad


def urdf_writer(
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
    output_path="output.urdf",
    type="finger",
):
    roll_B_X1_A = np.pi - angle_B_X1_A
    roll_X1_A_C = np.pi - angle_X1_A_C
    angle_B_X1_X2 = np.pi - (angle_B_X1_A + angle_A_X1_X2)
    roll_B_X1_X2 = angle_B_X1_X2

    width = 0.005
    link_z = 0.0001
    # # finger
    # if type == "finger":
    #     link_x1_A_y = 0.007
    #     # link_x1_B_y = 0.015
    #     # link_X2_C_y = 0.007 if link_X2_C_y is None else link_X2_C_y
    # else:
    #     link_x1_A_y = 0.015
    #     link_x1_B_y = 0.025
    #     link_X2_C_y = 0.015 if link_X2_C_y is None else link_X2_C_y

    # TODO: Get expression for point A, C, X2
    A = np.array(
        [
            -(link_x1_B_y - link_x1_A_y * np.cos(angle_B_X1_A)),
            -(link_x1_A_y * np.sin(angle_B_X1_A)),
        ]
    )
    support_angle_AC = angle_X1_A_C - angle_B_X1_A
    C = np.array(
        [
            A[0] - link_A_C_y * np.cos(support_angle_AC),
            A[1] - link_A_C_y * np.sin(support_angle_AC),
        ]
    )
    X1 = np.array([-link_x1_B_y, 0])
    support_angle_X1X2 = angle_B_X1_X2
    X2 = np.array(
        [
            X1[0] - link_X1_X2_y * np.cos(support_angle_X1X2),
            X1[1] - link_X1_X2_y * np.sin(support_angle_X1X2),
        ]
    )
    # print(support_angle_X1X2, A, C, X2)
    angle_X1_X2_C = angle_between_lines(X2, C, X1)
    # print(f"The angle between X1-X2 and X2-C is: {angle_X1_X2_C}")
    robot = ET.Element("robot", name="robot")
    link_base = create_link(
        robot=robot,
        name="link_base",
        size=f"{width} 0.00001 {link_z}",
        xyz=f"0 0 0",
        rpy="0 0 0",
        rgba="1.0 1.0 1.0 1.0",
    )
    link_x1_B = create_link(
        robot=robot,
        name="link_x1_B",
        size=f"{width} {link_x1_B_y} {link_z}",
        xyz=f"0 {-link_x1_B_y / 2} 0",
        rpy="0 0 0",
        rgba="1.0 0.0 0.0 1.0",
    )
    joint_base_x1 = create_joint(
        robot=robot,
        name="joint_base_x1",
        # joint_type="revolute",
        joint_type="fixed",
        parent_link="link_base",
        child_link="link_x1_B",
        origin_xyz="0 -0.00 0",
        origin_rpy="0 0 0",
        axis_xyz="1 0 0",
    )
    link_x1_A = create_link(
        robot=robot,
        name="link_x1_A",
        size=f"{width} {link_x1_A_y} {link_z}",
        xyz=f"0 {-link_x1_A_y / 2} 0",
        rpy="0 0 0",
        rgba="0.0 1.0 0.0 1.0",
    )
    joint_x1_A = create_joint(
        robot=robot,
        name="joint_x1_A",
        joint_type="fixed",
        parent_link="link_x1_B",
        child_link="link_x1_A",
        origin_xyz=f"0 {-link_x1_B_y} 0",
        origin_rpy=f"{roll_B_X1_A} 0 0",
        axis_xyz="1 0 0",
    )
    link_A_C = create_link(
        robot=robot,
        name="link_A_C",
        size=f"{width} {link_A_C_y} {link_z}",
        xyz=f"0 {-link_A_C_y / 2} 0",
        rpy="0 0 0",
        rgba="0.0 0.0 1.0 1.0",
    )
    joint_AC = create_joint(
        robot=robot,
        name="joint_AC",
        joint_type="revolute",
        parent_link="link_x1_A",
        child_link="link_A_C",
        origin_xyz=f"0 {-link_x1_A_y} 0",
        origin_rpy=f"{-roll_X1_A_C} 0 0",
        axis_xyz="1 0 0",
        lower=0,
        upper=np.pi,
    )
    link_X1_X2 = create_link(
        robot=robot,
        name="link_X1_X2",
        size=f"{width} {link_X1_X2_y} {link_z}",
        xyz=f"0 {-link_X1_X2_y / 2} 0",
        rpy="0 0 0",
        rgba="0.2 0.0 0.3 1.0",
    )
    joint_X1_X2 = create_joint(
        robot=robot,
        name="joint_X1_X2",
        joint_type="revolute",
        parent_link="link_x1_B",
        child_link="link_X1_X2",
        origin_xyz=f"{width} {-link_x1_B_y} 0",
        origin_rpy=f"{roll_B_X1_X2} 0 0",
        axis_xyz="1 0 0",
        lower=0,
        upper=np.pi,
    )
    link_X2_C = create_link(
        robot=robot,
        name="link_X2_C",
        size=f"{width} {link_X2_C_y} {link_z}",
        xyz=f"{0} {-link_X2_C_y / 2} 0",
        rpy="0 0 0",
        rgba="0.2 0.3 0.0 1.0",
    )
    joint_X2_C = create_joint(
        robot=robot,
        name="joint_X2_C",
        joint_type="revolute",
        parent_link="link_X1_X2",
        child_link="link_X2_C",
        origin_xyz=f"0 {-link_X1_X2_y} 0",
        origin_rpy=f"{-np.pi + angle_X1_X2_C} 0 0",
        axis_xyz="1 0 0",
        lower=0,
        upper=np.pi,
    )
    link_C_D = create_link(
        robot=robot,
        name="link_C_D",
        size=f"{width} {link_C_D_y} {link_z}",
        xyz=f"0 {-link_C_D_y / 2} 0",
        rpy="0 0 0",
        rgba="0.0 0.2 0.3 1.0",
    )
    joint_CD_CX2 = create_joint(
        robot=robot,
        name="joint_CD_CX2",
        joint_type="fixed",
        parent_link="link_X2_C",
        child_link="link_C_D",
        origin_xyz=f"{-width}  {-link_X2_C_y} 0",
        origin_rpy=f"{angle_X2_C_D}  0 0",
        # origin_rpy=f"{0}  0 0",
        axis_xyz="1 0 0",
    )

    closing_link_1 = create_link(
        robot=robot,
        name="closing_link_1",
        size=f"{0.002} {0.002} {0.002}",
        xyz=f"0 0 0",
        rpy=f"0 0 0",
        rgba="0.2 0.2 0.2 1.0",
    )
    closing_joint_1 = create_joint(
        robot=robot,
        name="new_closing_joint_1",
        joint_type="fixed",
        parent_link="link_A_C",
        child_link="closing_link_1",
        origin_xyz=f"0 {-link_A_C_y} 0",
        origin_rpy=f"0 0 0",
        axis_xyz="1 0 0",
    )
    closing_link_2 = create_link(
        robot=robot,
        name="closing_link_2",
        size=f"{0.002} {0.002} {0.002}",
        xyz=f"0 0 0",
        rpy=f"0 0 0",
        rgba="0.9 0.6 0.2 1.0",
    )
    closing_joint_2 = create_joint(
        robot=robot,
        name="new_closing_joint_2",
        joint_type="fixed",
        parent_link="link_X2_C",
        child_link="closing_link_2",
        origin_xyz=f"0 {-link_X2_C_y} {0}",
        origin_rpy=f"0 0 0",
        axis_xyz="1 0 0",
    )
    cube_dim = 0.01
    new_fingertips_link = create_link(
        robot=robot,
        name="new_fingertips_link",
        size=f"{cube_dim} {cube_dim} {cube_dim}",
        xyz=f"0 0 0",
        rpy=f"0 0 0",
        rgba="0.1 0.3 0.5 1.0",
    )
    joint_close_finger = create_joint(
        robot=robot,
        name="joint_close_finger",
        joint_type="fixed",
        parent_link="link_C_D",
        child_link="new_fingertips_link",
        origin_xyz=f"0 {-link_C_D_y + 0.009} 0"
        if type == "finger"
        else f"0 {-link_C_D_y + 0.011} 0",
        origin_rpy=f"{np.pi} {np.pi / 2} {0}",
        axis_xyz="1 0 0",
    )
    # Convert the ElementTree to a string
    xml_str = ET.tostring(robot, encoding="unicode")

    # Use minidom to prettify the XML
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")

    # Write the prettified XML to a file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)


# Print the prettified XML
# print(pretty_xml)


if __name__ == "__main__":
    urdf_writer(
        link_C_D_y=0.049500000000000016,
        link_X1_X2_y=0.037,
        link_A_C_y=0.036,
        link_X2_C_y=0.007,
        link_x1_A_y=0.007,
        link_x1_B_y=0.01895,
        angle_B_X1_A=0.7853981633974483,
        angle_X1_A_C=0.7853981633974483,
        angle_A_X1_X2=2.0616701789183027,
        angle_X2_C_D=2.454369260617026,
    )
