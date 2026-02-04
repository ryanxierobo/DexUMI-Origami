import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET

import numpy as np


def create_link(robot, name, size, xyz, rpy, rgba):
    link = ET.SubElement(robot, "link", name=name)
    visual = ET.SubElement(link, "visual")
    geometry = ET.SubElement(visual, "geometry")
    box = ET.SubElement(geometry, "box", size=size)
    origin = ET.SubElement(visual, "origin", rpy=rpy, xyz=xyz)
    material = ET.SubElement(visual, "material", name=f"{name}_material")
    color = ET.SubElement(material, "color", rgba=rgba)
    return link


def create_joint(
    robot,
    name,
    joint_type,
    parent_link,
    child_link,
    origin_xyz,
    origin_rpy,
    axis_xyz,
    lower=None,
    upper=None,
    effort=1,
    velocity=20,
):
    joint = ET.SubElement(robot, "joint", name=name, type=joint_type)
    parent = ET.SubElement(joint, "parent", link=parent_link)
    child = ET.SubElement(joint, "child", link=child_link)
    origin = ET.SubElement(joint, "origin", xyz=origin_xyz, rpy=origin_rpy)
    axis = ET.SubElement(joint, "axis", xyz=axis_xyz)
    if (
        lower is not None
        or upper is not None
        or effort is not None
        or velocity is not None
    ):
        limit = ET.SubElement(joint, "limit")
        if lower is not None:
            limit.set("lower", str(lower))
        if upper is not None:
            limit.set("upper", str(upper))
        limit.set("effort", str(effort))
        limit.set("velocity", str(velocity))
        return joint


def update_link_and_joint_names_with_prefix(robot, prefix):
    # Add prefix to all link names and store old->new name mapping
    link_name_mapping = {}
    for link in robot.findall(".//link"):
        old_name = link.get("name")
        if not old_name.startswith(prefix):
            new_name = f"{prefix}{old_name}"
            link_name_mapping[old_name] = new_name
            link.set("name", new_name)

    # Update all joint references using the mapping
    for joint in robot.findall(".//joint"):
        # Update joint name
        old_joint_name = joint.get("name")
        if not old_joint_name.startswith(prefix):
            joint.set("name", f"{prefix}{old_joint_name}")

        # Update parent link reference
        parent = joint.find("parent")
        old_parent = parent.get("link")
        if old_parent in link_name_mapping:
            parent.set("link", link_name_mapping[old_parent])

        # Update child link reference
        child = joint.find("child")
        old_child = child.get("link")
        if old_child in link_name_mapping:
            child.set("link", link_name_mapping[old_child])

    return link_name_mapping


def modify_link_parameter(robot, link_name, parameter_type, value):
    """
    Modify parameters of a link in a URDF robot description.

    Args:
        robot (xml.etree.ElementTree.Element): Root element of the URDF XML tree
        link_name (str): Name of the link to modify
        parameter_type (str): Type of parameter to modify ('xyz', 'rpy', 'size', 'rgba')
        value (str): New value for the parameter

    Returns:
        bool: True if modification was successful, False otherwise
    """
    # Find the link with the specified name
    link = robot.find(f".//link[@name='{link_name}']")
    if link is None:
        print(f"Warning: Link '{link_name}' not found")
        return False

    try:
        if parameter_type in ["xyz", "rpy"]:
            # These parameters are attributes of the origin element
            origin = link.find(".//origin")
            if origin is not None:
                origin.set(parameter_type, value)
            else:
                # If origin doesn't exist, create it under visual
                visual = link.find("visual")
                if visual is not None:
                    ET.SubElement(visual, "origin", **{parameter_type: value})
                else:
                    print(f"Warning: Visual element not found in link '{link_name}'")
                    return False

        elif parameter_type == "size":
            # Size is an attribute of the box element
            box = link.find(".//box")
            if box is not None:
                box.set("size", value)
            else:
                print(f"Warning: Box element not found in link '{link_name}'")
                return False

        elif parameter_type == "rgba":
            # RGBA is an attribute of the color element
            color = link.find(".//color")
            if color is not None:
                color.set("rgba", value)
            else:
                # If color doesn't exist, create it under material
                material = link.find(".//material")
                if material is not None:
                    ET.SubElement(material, "color", rgba=value)
                else:
                    print(f"Warning: Material element not found in link '{link_name}'")
                    return False

        else:
            print(f"Warning: Unsupported parameter type '{parameter_type}'")
            return False

        return True

    except Exception as e:
        print(f"Error modifying link parameter: {str(e)}")
        return False


def modify_joint_parameter(robot, joint_name, parameter_type, value):
    """
    Modify parameters of a joint in a URDF robot description.

    Args:
        robot (xml.etree.ElementTree.Element): Root element of the URDF XML tree
        joint_name (str): Name of the joint to modify
        parameter_type (str): Type of parameter to modify ('xyz', 'rpy', 'axis_xyz',
                            'lower', 'upper', 'effort', 'velocity')
        value (str or float): New value for the parameter

    Returns:
        bool: True if modification was successful, False otherwise
    """
    # Find the joint with the specified name
    joint = robot.find(f".//joint[@name='{joint_name}']")
    if joint is None:
        print(f"Warning: Joint '{joint_name}' not found")
        return False

    try:
        if parameter_type in ["xyz", "rpy"]:
            # These parameters are attributes of the origin element
            origin = joint.find("origin")
            if origin is not None:
                origin.set(parameter_type, str(value))
            else:
                # If origin doesn't exist, create it
                ET.SubElement(joint, "origin", **{parameter_type: str(value)})

        elif parameter_type == "axis_xyz":
            # Modify the axis xyz attribute
            axis = joint.find("axis")
            if axis is not None:
                axis.set("xyz", str(value))
            else:
                # If axis doesn't exist, create it
                ET.SubElement(joint, "axis", xyz=str(value))

        elif parameter_type in ["lower", "upper", "effort", "velocity"]:
            # These parameters are attributes of the limit element
            limit = joint.find("limit")
            if limit is not None:
                limit.set(parameter_type, str(value))
            else:
                # If limit doesn't exist, create it with the specified parameter
                limit_attrs = {parameter_type: str(value)}

                # Add default values for required attributes if creating new limit element
                if parameter_type in ["lower", "upper"]:
                    if "effort" not in limit_attrs:
                        limit_attrs["effort"] = "1"
                    if "velocity" not in limit_attrs:
                        limit_attrs["velocity"] = "20"

                ET.SubElement(joint, "limit", **limit_attrs)

        elif parameter_type == "type":
            # Modify the joint type
            joint.set("type", str(value))

        else:
            print(f"Warning: Unsupported parameter type '{parameter_type}'")
            return False

        return True

    except Exception as e:
        print(f"Error modifying joint parameter: {str(e)}")
        return False
