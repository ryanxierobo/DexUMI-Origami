import os
import time

import click
import cv2
import numpy as np
import scipy.spatial.transform as st
import zarr
from matplotlib import pyplot as plt

from dexumi.camera.camera import FrameData
from dexumi.camera.iphone_camera import IphoneCamera
from dexumi.common.precise_sleep import precise_wait
from dexumi.common.utility.matrix import (
    invert_transformation,
    relative_transformation,
    visualize_multiple_frames_and_points,
)
from dexumi.constants import (
    INSPIRE_PER_FINGER_ADJ_VAL,
    INSPIRE_POS_READ_RATE,
)
from dexumi.hand_sdk.inspire.hand_api_cls import ExoInspireSDK
from dexumi.hand_sdk.xhand.hand_api_cls import ExoXhandSDK
from dexumi.real_env.common.ur5 import UR5eClient, UR5Server
from dexumi.real_env.spacemouse import Spacemouse
from real_script.data_generation_pipeline.camera_position import (
    get_camera_position_and_orientation,
)

hand_id = 0x01  # Hand ID


x_offset = -0.0395
y_offset = -0.1342
z_offset = 0.0428
MARKER_SIZE = (155 + 32.5) / 1000


@click.command()
@click.option("-rh", "--robot_hostname", default="192.168.1.102")
@click.option("-f", "--frequency", type=float, default=10)
@click.option("-e", "--episode", type=int, default=0)
@click.option("--hand-port", default="/dev/ttyUSB0", help="Hand Port", type=str)
@click.option(
    "--finger-mapping-model-path",
    type=click.Path(),
    help="Path to the finger mapping model",
)
@click.option(
    "--hand-type",
    type=click.Choice(["inspire", "xhand"]),
    default="inspire",
    help="Type of hand to use",
)
@click.option(
    "-r",
    "--replay_path",
)
def main(
    robot_hostname,
    frequency,
    replay_path,
    episode,
    finger_mapping_model_path,
    hand_port,
    hand_type,
):
    hand_type = "xhand"
    max_pos_speed = 0.3
    max_rot_speed = 0.6
    tcp_offset = 0.1
    ur5_server = UR5Server(
        robot_ip=robot_hostname,
        pub_frequency=100,
        req_frequency=500,
        max_buffer_size=2000,
        frames_per_publish=200,
        frequency=500,
        lookahead_time=0.05,
        gain=1000,
        max_pos_speed=0.25,
        max_rot_speed=0.6,
        tcp_offset_pose=[0, 0, tcp_offset, 0, 0, 0],
        verbose=False,
    )
    ur5_server.start()
    robot_client = UR5eClient()
    if hand_type == "xhand":
        per_finger_adj_val = [0] * 12
        per_finger_adj_val[0] = 4.5
        per_finger_adj_val[1] = -4.8
        per_finger_adj_val[4] = -1
        per_finger_adj_val[6] = 2
        per_finger_adj_val[8] = 3.5
        per_finger_adj_val[10] = 4

        hand = ExoXhandSDK(
            port=hand_port,
            calibration_dir=finger_mapping_model_path,
            per_finger_adj_val=per_finger_adj_val,
        )
    elif hand_type == "inspire":
        hand = ExoInspireSDK(
            port=hand_port,
            finger_mapping_model_path=os.path.join(
                finger_mapping_model_path, "joint_to_motor_index.pkl"
            ),
            thumb_swing_model_path=os.path.join(
                finger_mapping_model_path, "joint_to_motor_thumb_swing.pkl"
            ),
            thumb_middle_model_path=os.path.join(
                finger_mapping_model_path, "joint_to_motor_thumb_proximal_0.pkl"
            ),
            per_finger_ref_val=INSPIRE_PER_FINGER_ADJ_VAL,  # value incr to make finger ahead
            read_rate=INSPIRE_POS_READ_RATE,
        )
    if hand.connect():
        # hand.start_reader()
        print("hand ready!")
    app = IphoneCamera()
    app.start_streaming()
    # A be the qr code frame
    # B be the robot frame
    # C be the camera frame
    # T_AB = T_AC * T_CB
    replay_buffer = zarr.open(replay_path, mode="r")
    episode_data = replay_buffer[f"episode_{episode}"]
    rgb = episode_data["replay_rgb"][:]
    depth = episode_data["replay_depth"][:]
    plt.imshow(rgb)
    plt.show()
    plt.imshow(depth)
    plt.show()
    pose = episode_data["pose_interp"][:]
    intrinsics = episode_data["replay_intrinsic"][:]
    joint_values = episode_data["joint_angles_interp"][:]
    print(pose[0])
    visualize_multiple_frames_and_points(frames_dict={"pose": pose}, show_axes=True)
    relative_pose = np.array([relative_transformation(pose[0], p) for p in pose])
    initial_frame_data = FrameData(
        rgb=rgb,
        depth=depth,
        pose=pose[0],
        intrinsics=intrinsics,
    )
    T_AC, rgb_image = get_camera_position_and_orientation(
        initial_frame_data, (155 + 32.5) / 1000, viz=False
    )

    plt.imshow(rgb_image)
    plt.show()
    with (
        Spacemouse() as sm,
    ):
        print("Ready!")
        dt = 1 / frequency
        command_latency = dt / 2
        state = robot_client.get_state().state
        target_pose = state["TargetTCPPose"]
        t_start = time.monotonic()

        iter_idx = 0
        while True:
            s = time.time()
            t_cycle_end = t_start + (iter_idx + 1) * dt
            t_sample = t_cycle_end - command_latency
            t_command_target = t_cycle_end + dt

            precise_wait(t_sample)
            sm_state = sm.get_motion_state_transformed()
            # print(sm_state)
            dpos = sm_state[:3] * (max_pos_speed / frequency)
            drot_xyz = sm_state[3:] * (max_rot_speed / frequency)

            drot = st.Rotation.from_euler("xyz", drot_xyz)
            # dpos[[1, 2]] = 0
            target_pose[:3] += dpos
            target_pose[3:] = (
                drot * st.Rotation.from_rotvec(target_pose[3:])
            ).as_rotvec()
            dpos = 0
            frame_data = app.get_camera_frame()
            cv2.imshow("RGB", frame_data.rgb)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            robot_client.schedule_waypoint(
                target_pose, t_command_target - time.monotonic() + time.time()
            )

            precise_wait(t_cycle_end)
            iter_idx += 1

        T_AD, rgb_image = get_camera_position_and_orientation(frame_data, MARKER_SIZE)
        state = robot_client.get_state().state
        end_pose = state["ActualTCPPose"]
        T_BE = np.eye(4)
        T_BE[:3, :3] = st.Rotation.from_rotvec(end_pose[3:]).as_matrix()
        T_BE[:3, -1] = end_pose[:3]
        T_BE[3, 3] = 1
        T_ED = np.array(
            [
                [0, 1, 0, x_offset],
                [-1, 0, 0, y_offset],
                [0, 0, 1, z_offset],
                [0, 0, 0, 1],
            ]
        )
        # (y,-x,z)
        T_BD = T_BE @ T_ED
        T_CA = invert_transformation(T_AC)
        T_CD = T_CA @ T_AD
        T_DC = invert_transformation(T_CD)
        T_DB = invert_transformation(T_BD)
        # from end to base (change of coordinate) => base to camera (change of coordinate) => camera transformation to new frame (apply rotation) => tranform to base(change of coordinate)
        replay_target = T_BD @ T_DC @ T_DB @ T_BE
        print("replay_target:\n", replay_target)
        t_start = time.monotonic()

        iter_idx = 0
        while True:
            current_pose = state["ActualTCPPose"]
            target_pose = state["ActualTCPPose"]
            s = time.time()
            t_cycle_end = t_start + (iter_idx + 1) * dt
            t_sample = t_cycle_end - command_latency
            t_command_target = t_cycle_end + dt

            precise_wait(t_sample)
            dpos = (replay_target[:3, -1] - current_pose[:3]) * (
                max_pos_speed / frequency
            )
            # T_BE@T_delta@T_EB
            R_BE = st.Rotation.from_rotvec(current_pose[3:]).as_matrix()
            drot_matrix = R_BE.T @ replay_target[:3, :3]
            drot_xyz = st.Rotation.from_matrix(drot_matrix).as_euler("xyz") * (
                max_rot_speed / frequency
            )
            drot_matrix = st.Rotation.from_euler("xyz", drot_xyz).as_matrix()
            target_pose[:3] += dpos
            target_pose[3:] = st.Rotation.from_matrix(
                (R_BE @ drot_matrix @ R_BE.T @ R_BE)
            ).as_rotvec()
            # target_pose[:3] = np.clip(
            #     target_pose[:3],
            #     np.array([-0.8, -0.4, 0.2]),
            #     np.array([-0.4, 0.5, 0.6]),
            # )
            dpos = 0
            frame_data = app.get_camera_frame()
            cv2.imshow("RGB", frame_data.rgb)
            cv2.imshow("RGB (Initial)", rgb_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            robot_client.schedule_waypoint(
                target_pose, t_command_target - time.monotonic() + time.time()
            )

            precise_wait(t_cycle_end)
            iter_idx += 1
        ######################## replay start ############################
        t_start = time.monotonic()
        T_ET = np.array(
            [
                [0, -1, 0, x_offset],
                [-1, 0, 0, y_offset],
                [0, 0, -1, z_offset],
                [0, 0, 0, 1],
            ]
        )
        ur5_replay_initial_pose = state["ActualTCPPose"]
        T_BE = np.eye(4)
        T_BE[:3, :3] = st.Rotation.from_rotvec(ur5_replay_initial_pose[3:]).as_matrix()
        T_BE[:3, -1] = ur5_replay_initial_pose[:3]
        # save T_BE as npy
        # exit(0)
        print(relative_pose.shape)
        print(pose.shape)
        T_BN = np.zeros_like(relative_pose)
        for iter_idx in range(len(relative_pose)):
            T_BN[iter_idx] = (
                T_BE @ T_ET @ relative_pose[iter_idx] @ invert_transformation(T_ET)
            )
        visualize_multiple_frames_and_points(frames_dict={"T_BN": T_BN}, show_axes=True)

        iter_idx = 0
        while True:
            target_pose = state["ActualTCPPose"]
            t_cycle_end = t_start + (iter_idx + 1) * dt
            t_command_target = t_cycle_end + dt
            target_pose[:3] = T_BN[iter_idx, :3, -1]
            target_pose[3:] = st.Rotation.from_matrix(
                T_BN[iter_idx, :3, :3]
            ).as_rotvec()
            # target_pose[:3] = np.clip(
            #     target_pose[:3],
            #     np.array([-1.0, -1, 0.01]),
            #     np.array([1.0, 1, 0.7]),
            # )
            dpos = 0
            frame_data = app.get_camera_frame()
            cv2.imshow("RGB", frame_data.rgb)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            joint_angles = (joint_values[iter_idx]).astype(int)
            predicted_motor_value = hand.predict_motor_value(joint_angles)
            print("predicted_motor_value", predicted_motor_value)
            command = hand.write_hand_angle_position_from_motor(predicted_motor_value)
            hand.send_command(command)
            robot_client.schedule_waypoint(
                target_pose, t_command_target - time.monotonic() + time.time()
            )

            precise_wait(t_cycle_end)
            iter_idx += 1


if __name__ == "__main__":
    main()
