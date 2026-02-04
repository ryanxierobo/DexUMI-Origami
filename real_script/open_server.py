import os
import time

import click

from dexumi.camera.oak_camera import OakCamera, get_all_oak_cameras
from dexumi.constants import (
    INSPIRE_PER_FINGER_ADJ_VAL,
    INSPIRE_POS_READ_RATE,
)
from dexumi.hand_sdk.inspire.hand_api_cls import ExoInspireSDK
from dexumi.hand_sdk.xhand.hand_api_cls import ExoXhandSDK
from dexumi.real_env.common.camera import CameraServer
from dexumi.real_env.common.dexhand import DexServer
from dexumi.real_env.common.ur5 import UR5Server


@click.command()
@click.option("--protocol", type=click.Choice(["ipc", "tcp"]), default="ipc")
@click.option("--camera", is_flag=True)
@click.option("--dexhand", is_flag=True)
@click.option("--ur5", is_flag=True)
@click.option("-rh", "--robot_hostname", default="192.168.1.102")
@click.option(
    "--hand-type",
    type=click.Choice(["inspire", "xhand"]),
    default="inspire",
    help="Type of hand to use",
)
@click.option(
    "--finger-mapping-model-path",
    type=click.Path(),
    help="Path to the finger mapping model",
)
def main(
    protocol, camera, dexhand, ur5, robot_hostname, hand_type, finger_mapping_model_path
):
    # Set base address based on protocol
    base_addr = "ipc:///tmp/" if protocol == "ipc" else "tcp://*:"

    if camera:
        connected_oak_cameras = get_all_oak_cameras()
        cameras = []
        for camera in connected_oak_cameras:
            oak_camera = OakCamera("oak", device_id=camera)
            cameras.append(oak_camera)
        camera_server = CameraServer(
            camera=cameras[1],
            pub_address=f"{base_addr}{'camera_stream' if protocol == 'ipc' else '5555'}",
            req_address=f"{base_addr}{'camera_req' if protocol == 'ipc' else '5556'}",
            max_buffer_size=30,
            pub_frequency=30,
            frames_per_publish=1,
            compression="jpeg",
            compression_quality=30,
        )
        camera_server.start()

    if dexhand:
        if hand_type == "xhand":
            per_finger_adj_val = [0] * 12
            per_finger_adj_val[0] = 4.5
            per_finger_adj_val[1] = -4.8
            per_finger_adj_val[4] = -1
            per_finger_adj_val[6] = 2
            per_finger_adj_val[8] = 3.5
            per_finger_adj_val[10] = 4

            hand = ExoXhandSDK(
                port="/dev/ttyUSB0",
                calibration_dir=finger_mapping_model_path,
                per_finger_adj_val=per_finger_adj_val,
            )
        elif hand_type == "inspire":
            hand = ExoInspireSDK(
                port="/dev/ttyUSB0",
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
        hand.connect()
        time.sleep(1)
        hand.start_reader()
        dex_server = DexServer(
            inspire=hand,
            pub_address=f"{base_addr}{'dex_stream' if protocol == 'ipc' else '5559'}",
            req_address=f"{base_addr}{'dex_req' if protocol == 'ipc' else '5560'}",
            max_buffer_size=20,
            pub_frequency=10,
            topic="dexhand",
            frequency=30,
            verbose=False,
        )
        dex_server.start()

    if ur5:
        ur5_server = UR5Server(
            robot_ip=robot_hostname,
            pub_frequency=100,
            req_frequency=500,
            max_buffer_size=2000,
            frames_per_publish=200,
            # max_buffer_size=2000,
            # frames_per_publish=1000,
            frequency=500,
            lookahead_time=0.05,
            gain=1000,
            max_pos_speed=0.25,
            max_rot_speed=0.6,
            # tcp_offset_pose=[0, 0, 0, 0, 0, 0],
            tcp_offset_pose=[0, 0, 0.10, 0, 0, 0],
            verbose=False,
        )
        ur5_server.start()


if __name__ == "__main__":
    main()
