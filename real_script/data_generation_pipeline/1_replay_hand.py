import os
import shutil
import sys
import time

import click
import cv2
import numpy as np
import zarr

from dexumi.camera.oak_camera import OakCamera, get_all_oak_cameras
from dexumi.common.frame_manager import FrameRateContext
from dexumi.constants import (
    INSPIRE_PER_FINGER_ADJ_VAL,
    INSPIRE_POS_READ_RATE,
    INSPIRE_SEG_VAL,
    XHAND_SEG_VAL,
)
from dexumi.hand_sdk.inspire.hand_api_cls import ExoInspireSDK
from dexumi.hand_sdk.xhand.hand_api_cls import ExoXhandSDK

# Get the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Calculate relative paths
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CONSTANTS_PATH = os.path.join(REPO_ROOT, "dexumi", "constants.py")


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(),
    default="~/Dev/DexUMI/data_local/exoskeleton_recordings",
    help="Base directory for storing recorded data",
)
@click.option(
    "--save-dir",
    type=click.Path(),
    default="~/Dev/DexUMI/data_local/exoskeleton_replay",
    help="Base directory for storing replayed data",
)
@click.option(
    "--episode_index", "-e", type=int, default=0, help="Index of the episode to replay"
)
# @click.option("--hand-id", default=0x01, help="Hand ID (hex value)", type=int)
@click.option("--hand-port", default="/dev/ttyUSB0", help="Hand Port", type=str)
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
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
@click.option("--headless", is_flag=True, help="Enable headless")
@click.option(
    "--fps", type=float, default=4, help="Frame rate for replaying the recording"
)
@click.option(
    "--camera_latency", type=float, default=0.1849, help="Camera latency in seconds"
)
@click.option(
    "--reference-dir",
    default=None,
    help="Base directory for reference data",
)
def replay_hand(
    data_dir,
    save_dir,
    episode_index,
    hand_port,
    hand_type,
    finger_mapping_model_path,
    verbose,
    headless,
    fps,
    camera_latency,
    reference_dir=None,
):
    data_dir = os.path.expanduser(data_dir)
    data_buffer = zarr.open(data_dir, mode="a")
    episode_data = data_buffer[f"episode_{episode_index}"]
    encoder_data = episode_data["numeric_0"]
    joint_angles_interp = encoder_data["joint_angles_interp"][:]
    if verbose:
        click.echo(f"Joint_angles interp shape: {joint_angles_interp.shape}")
    save_dir = os.path.expanduser(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    if reference_dir is not None:
        # First, copy reference directory contents to save directory
        reference_name = os.path.basename(reference_dir)
        target_dir = os.path.join(save_dir, reference_name)

        # Create the target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)

        # Copy reference directory contents
        os.system(f"cp -r {reference_dir}/* {target_dir}/")

        try:
            # Copy constants.py to the new location in save_dir/reference_dir
            shutil.copy2(CONSTANTS_PATH, os.path.join(target_dir, "constants.py"))
            if verbose:
                click.echo(f"Copied constants.py to {target_dir}")
        except Exception as e:
            click.echo(f"Error copying constants.py: {e}", err=True)

    # Initialize data sources
    oak_cameras = get_all_oak_cameras()
    if not oak_cameras:
        click.echo("Error: No OAK cameras found", err=True)
        return

    camera = OakCamera("oak", device_id=oak_cameras[0])
    camera.start_streaming()

    # init hand
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
    hand.connect()
    max_iter = joint_angles_interp.shape[0]
    ######### move to segment position ########
    for i in range(10):
        with FrameRateContext(fps, verbose=verbose) as fr:
            frames = camera.get_camera_frame()
            if frames is not None:
                # Visualize frame
                viz_frame = frames.rgb.copy()
                if not headless:
                    cv2.imshow("Recording Preview", viz_frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        sys.exit()
                    elif key == ord("s"):
                        click.echo("Starting recording...")
                        break
            command = hand.write_hand_angle_position_from_motor(
                INSPIRE_SEG_VAL if hand_type == "inspire" else XHAND_SEG_VAL
            )
            hand.send_command(command)
    height, width = viz_frame.shape[:2]
    os.makedirs(os.path.join(save_dir, f"episode_{episode_index}"), exist_ok=True)
    video_writer = cv2.VideoWriter(  # Store in instance variable
        os.path.join(save_dir, f"episode_{episode_index}", "dex_camera_0.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (width, height),
    )
    # take the first image for segmentation
    time.sleep(camera_latency)
    frames = camera.get_camera_frame()
    video_writer.write(frames.rgb)
    iter_idx = 0
    ######### move to first frame position from replay ########
    for i in range(10):
        with FrameRateContext(fps, verbose=verbose) as fr:
            # frames = recorder_manager.get_latest_frames()
            frames = camera.get_camera_frame()
            if frames is not None:
                # Visualize frame
                viz_frame = frames.rgb.copy()
                if not headless:
                    cv2.imshow("Recording Preview", viz_frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        sys.exit()
                    elif key == ord("s"):
                        click.echo("Starting recording...")
                        break
                    else:
                        click.echo("Recording already started.")
            motor_value = hand.predict_motor_value(
                joint_angles_interp[iter_idx]
            )  # from thumb to pinky
            command = hand.write_hand_angle_position_from_motor(motor_value)
            hand.send_command(command)
    ######### Start replay and recording #########
    iter_idx = 0
    predicted_motor_values = []
    control_fps = 30
    while True:
        with FrameRateContext(fps, verbose=verbose) as fr:
            while True:
                with FrameRateContext(control_fps, verbose=verbose) as cfr:
                    # this frame is only for visualization
                    frames = camera.get_camera_frame()
                    if frames is not None:
                        # Visualize frame
                        viz_frame = frames.rgb.copy()
                        cv2.putText(
                            viz_frame,
                            f"Episode: {episode_index}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
                        if not headless:
                            cv2.imshow("Recording Preview", viz_frame)
                            key = cv2.waitKey(1) & 0xFF

                            if key == ord("q"):
                                break

                    motor_value = hand.predict_motor_value(
                        joint_angles_interp[iter_idx]
                    )  # from thumb to pinky
                    command = hand.write_hand_angle_position_from_motor(motor_value)
                    hand.send_command(command)
                    if fr.this_iter_end_time < time.monotonic():
                        print("break inner loop")
                        break
            iter_idx += 1

        predicted_motor_values.append(motor_value)
        # recapture the frame here to ensure the frame and motor value is aligned
        time.sleep(camera_latency)
        frames = camera.get_camera_frame()
        video_writer.write(frames.rgb)
        if iter_idx == max_iter:
            break
        print(
            f"\rProgress: {iter_idx}/{max_iter} ({iter_idx / max_iter * 100:.1f}%)",
            end="",
        )

    video_writer.release()
    cv2.destroyAllWindows()
    predicted_motor_values = np.array(predicted_motor_values)
    data_buffer = zarr.open(save_dir, mode="a")
    episode_data = data_buffer[f"episode_{episode_index}"]
    episode_data["hand_motor_value"] = predicted_motor_values


if __name__ == "__main__":
    replay_hand()
