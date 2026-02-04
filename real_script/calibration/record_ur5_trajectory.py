import copy
import time

import click
import cv2
import numpy as np
import scipy.spatial.transform as st
import zarr

from dexumi.camera.iphone_camera import IphoneCamera
from dexumi.common.precise_sleep import precise_wait
from dexumi.real_env.common.ur5 import UR5eClient, UR5Server


@click.command()
@click.option("-rh", "--robot_hostname", default="192.168.1.102")
@click.option("-f", "--frequency", type=float, default=45)
@click.option("-rp", "--record_path", type=str, default="iphone_calibration")
def main(robot_hostname, frequency, record_path):
    max_pos_speed = 0.05
    max_rot_speed = 0.2
    tcp_offset = 0.0
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
    camera = IphoneCamera()
    camera.start_streaming()
    dt = 1 / frequency
    if record_path:
        record_roots = zarr.open(record_path, mode="a")
        keys = list(record_roots.group_keys())
        episode_number = len(keys)
        episode_data = record_roots.require_group(f"episode_{episode_number}")
        print(f"episode_{episode_number}")
    rest_rotation_matrix = np.array(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1],
        ]
    )

    state = robot_client.get_state().state
    target_pose = state["TargetTCPPose"]
    t_start = time.monotonic()
    iter_idx = 0
    target_pose[:3] = np.array([-0.47284942, -0.01612812, 0.42368649])
    target_pose[3:] = st.Rotation.from_matrix(rest_rotation_matrix).as_rotvec()
    while True:
        t_cycle_end = t_start + (iter_idx + 1) * dt
        t_command_target = t_cycle_end + dt
        robot_client.schedule_waypoint(
            target_pose, t_command_target - time.monotonic() + time.time()
        )
        frame_data = camera.get_camera_frame()
        if frame_data.rgb is not None:
            cv2.imshow("RGB", frame_data.rgb)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        precise_wait(t_cycle_end)
        iter_idx += 1

    state = robot_client.get_state().state
    target_pose = state["TargetTCPPose"]
    t_start = time.monotonic()
    iter_idx = 0
    ee_pose_buffer = []
    rgb_buffer = []
    depth_buffer = []
    instrinsics_buffer = []
    iphone_pose_buffer = []
    # target_pose[:3] = np.array([-0.5, 0.07, 0.28])
    target_pose[3:] = (
        st.Rotation.from_euler("zyx", [0, 0, 0])
        * st.Rotation.from_matrix(rest_rotation_matrix)
    ).as_rotvec()
    pose_change_cycle = 4 * frequency
    while True:
        current_pose = state["ActualTCPPose"]
        print(current_pose)
        t_cycle_end = t_start + (iter_idx + 1) * dt
        t_command_target = t_cycle_end + dt
        if iter_idx % pose_change_cycle == 0:
            target_pose[3:] = (
                st.Rotation.from_euler(
                    "zyx",
                    [
                        np.random.uniform(low=-np.pi / 4, high=np.pi / 4),
                        np.random.uniform(low=-np.pi / 4, high=np.pi / 4),
                        np.random.uniform(low=-np.pi / 4, high=np.pi / 4),
                    ],
                )
                * st.Rotation.from_matrix(rest_rotation_matrix)
            ).as_rotvec()
        robot_client.schedule_waypoint(
            target_pose, t_command_target - time.monotonic() + time.time()
        )
        ee_pose_buffer.append(copy.deepcopy(current_pose))
        frame_data = camera.get_camera_frame()
        rgb_buffer.append(cv2.cvtColor(frame_data.rgb, cv2.COLOR_RGB2BGR))
        depth_buffer.append(frame_data.depth)
        instrinsics_buffer.append(frame_data.intrinsics)
        iphone_pose_buffer.append(frame_data.pose)
        print(frame_data.pose)

        if frame_data.rgb is not None:
            cv2.imshow("RGB", frame_data.rgb)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        precise_wait(t_cycle_end)
        iter_idx += 1

    if record_path:
        print("saving data")
        # episode_data["rgb"] = rgb_buffer
        # episode_data["depth"] = depth_buffer
        episode_data["intrinsics"] = instrinsics_buffer
        episode_data["iphone_pose"] = iphone_pose_buffer
        episode_data["ee_pose"] = ee_pose_buffer

    ur5_server.stop()


if __name__ == "__main__":
    main()
