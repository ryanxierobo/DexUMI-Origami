import time

import click
import cv2
import numpy as np
import scipy.spatial.transform as st

from dexumi.camera.camera import FrameData
from dexumi.camera.iphone_camera import IphoneCamera
from dexumi.common.precise_sleep import precise_wait
from dexumi.common.utility.matrix import (
    invert_transformation,
    relative_transformation,
)
from dexumi.data_recording.record_manager import RecorderManager
from dexumi.data_recording.video_recorder import VideoRecorder
from dexumi.encoder.encoder import XhandEncoder
from dexumi.hand_sdk.xhand.hand_api_cls import ExoXhandSDK
from dexumi.real_env.common.ur5 import UR5eClient, UR5Server
from dexumi.real_env.spacemouse import Spacemouse

x_offset = -0.0395
y_offset = -0.1342
z_offset = 0.0428

T_ET = np.array(
    [
        [0, -1, 0, x_offset],
        [-1, 0, 0, y_offset],
        [0, 0, -1, z_offset],
        [0, 0, 0, 1],
    ]
)
joint_value_to_motor_scalar = 1000


@click.command()
@click.option("-rh", "--robot_hostname", default="192.168.1.102")
@click.option("-f", "--frequency", type=float, default=30)
@click.option(
    "--finger-mapping-model-path",
    type=click.Path(),
    help="Path to the finger mapping model",
)
def main(robot_hostname, frequency, finger_mapping_model_path):
    max_pos_speed = 0.3
    max_rot_speed = 0.15
    tcp_offset = 0.0
    dt = 1 / frequency
    fps = 45
    ur5_server = UR5Server(
        robot_ip=robot_hostname,
        pub_frequency=100,
        req_frequency=500,
        max_buffer_size=2000,
        frames_per_publish=200,
        frequency=500,
        lookahead_time=0.05,
        gain=1000,
        max_pos_speed=max_pos_speed,
        max_rot_speed=max_rot_speed,
        tcp_offset_pose=[0, 0, tcp_offset, 0, 0, 0],
        verbose=False,
    )
    ur5_server.start()
    robot_client = UR5eClient()

    # Initialize the camera
    camera_sources = [
        IphoneCamera(),
    ]
    data_dir = "teleop"
    video_recorder = VideoRecorder(
        record_fps=fps,
        stream_fps=60,
        video_record_path=data_dir,
        camera_sources=camera_sources,
        frame_data_class=FrameData,
        # verbose=verbose,
        verbose=False,
    )
    recorder_manager = RecorderManager(
        recorders=[
            video_recorder,
        ],
        verbose=False,
    )
    if not recorder_manager.start_streaming():
        click.echo("Failed to start streaming", err=True)
        return

    per_finger_adj_val = [0] * 12
    hand = ExoXhandSDK(
        port="/dev/ttyUSB0",
        calibration_dir=finger_mapping_model_path,
        per_finger_adj_val=per_finger_adj_val,
    )
    hand.connect()
    dexhand_encoder = XhandEncoder("inspire", verbose=False, uart_port="/dev/ttyACM0")
    dexhand_encoder.start_streaming()
    with (
        Spacemouse() as sm,
    ):
        state = robot_client.get_state().state
        reset_pose = state["TargetTCPPose"]
        while True:
            iter_idx = 0
            t_start = time.monotonic()
            while True:
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_command_target = t_cycle_end + dt
                state = robot_client.get_state().state
                ur5_replay_initial_pose = state["ActualTCPPose"]
                T_BE = np.eye(4)
                T_BE[:3, :3] = st.Rotation.from_rotvec(
                    ur5_replay_initial_pose[3:]
                ).as_matrix()
                T_BE[:3, -1] = ur5_replay_initial_pose[:3]
                frames = recorder_manager.get_latest_frames()
                tracking_frame = frames["IphoneCamera"][-1]
                iphone_init_pose = tracking_frame.pose
                cv2.imshow("VIZ", tracking_frame.rgb)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    exit(0)
                robot_client.schedule_waypoint(
                    reset_pose, t_command_target - time.monotonic() + time.time()
                )
                precise_wait(t_cycle_end)

            iter_idx = 0
            t_start = time.monotonic()
            while True:
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_command_target = t_cycle_end + dt

                # precise_wait(t_sample)
                frames = recorder_manager.get_latest_frames()
                tracking_frame = frames["IphoneCamera"][-1]
                iphone_pose = tracking_frame.pose
                relative_pose = relative_transformation(iphone_init_pose, iphone_pose)
                T_BN = T_BE @ T_ET @ relative_pose @ invert_transformation(T_ET)
                target_pose = state["ActualTCPPose"]
                target_pose[:3] = T_BN[:3, -1]
                target_pose[3:] = st.Rotation.from_matrix(T_BN[:3, :3]).as_rotvec()
                target_pose[:3] = np.clip(
                    target_pose[:3],
                    np.array([-1.0, -1, 0.01]),
                    np.array([1.0, 1, 0.7]),
                )
                numeric_frame = dexhand_encoder.get_numeric_frame()
                joint_angles = numeric_frame.joint_angles
                predicted_motor_value = hand.predict_motor_value(joint_angles)
                command = hand.write_hand_angle_position_from_motor(
                    predicted_motor_value
                )
                hand.send_command(command)
                robot_client.schedule_waypoint(
                    target_pose, t_command_target - time.monotonic() + time.time()
                )
                cv2.imshow("VIZ", tracking_frame.rgb)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    exit(0)
                precise_wait(t_cycle_end)
                iter_idx += 1


if __name__ == "__main__":
    main()
