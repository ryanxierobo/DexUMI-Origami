import time

import click
import cv2
import numpy as np
import scipy.spatial.transform as st

from dexumi.common.precise_sleep import precise_wait
from dexumi.real_env.common.ur5 import UR5eClient, UR5Server
from dexumi.real_env.spacemouse import Spacemouse


@click.command()
@click.option("-rh", "--robot_hostname", default="192.168.1.102")
@click.option("-f", "--frequency", type=float, default=30)
def main(robot_hostname, frequency):
    max_pos_speed = 0.25
    max_rot_speed = 0.6
    tcp_offset = 0.0
    dt = 1 / frequency
    command_latency = dt / 2
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
    with (
        Spacemouse() as sm,
    ):
        print("Ready!")
        # to account for recever interfance latency, use target pose
        # to init buffer.
        state = robot_client.get_state().state
        target_pose = state["TargetTCPPose"]

        t_start = time.monotonic()

        iter_idx = 0
        while True:
            current_pose = state["ActualTCPPose"]
            t_cycle_end = t_start + (iter_idx + 1) * dt
            t_sample = t_cycle_end - command_latency
            t_command_target = t_cycle_end + dt

            precise_wait(t_sample)
            sm_state = sm.get_motion_state_transformed()
            # print(sm_state)
            dpos = sm_state[:3] * (max_pos_speed / frequency)
            drot_xyz = sm_state[3:] * (max_rot_speed / frequency)

            drot = st.Rotation.from_euler("xyz", drot_xyz)
            target_pose[:3] += dpos
            target_pose[3:] = (
                drot * st.Rotation.from_rotvec(target_pose[3:])  # T_BI*T_IE = T_BE
            ).as_rotvec()
            dpos = 0
            print(
                target_pose[:3],
                st.Rotation.from_rotvec(target_pose[3:]).as_matrix(),
            )
            robot_client.schedule_waypoint(
                target_pose, t_command_target - time.monotonic() + time.time()
            )
            precise_wait(t_cycle_end)
            iter_idx += 1
            random_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imshow("Random", random_img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    main()
