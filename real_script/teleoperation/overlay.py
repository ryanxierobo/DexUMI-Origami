import os
import time

import click
import cv2

from dexumi.camera.oak_camera import OakCamera, get_all_oak_cameras
from dexumi.common.frame_manager import FrameRateContext
from dexumi.constants import INSPIRE_PER_FINGER_ADJ_VAL, INSPIRE_POS_READ_RATE
from dexumi.encoder.encoder import InspireEncoder, XhandEncoder
from dexumi.hand_sdk.inspire.hand_api_cls import ExoInspireSDK
from dexumi.hand_sdk.xhand.hand_api_cls import ExoXhandSDK


@click.command()
@click.option(
    "--hand-type",
    type=click.Choice(["xhand", "inspire"]),
    default="xhand",
    help="Type of hand to control (xhand or inspire)",
)
@click.option("--hand-port", default="/dev/ttyUSB0", help="Hand Port", type=str)
@click.option(
    "--finger-mapping-model-path",
    type=click.Path(),
    help="Path to the finger mapping model",
)
def overlay(hand_type, hand_port, finger_mapping_model_path):
    connected_oak_cameras = get_all_oak_cameras()
    cameras = []
    for camera in connected_oak_cameras:
        oak_camera = OakCamera("oak", device_id=camera)
        oak_camera.start_streaming()
        cameras.append(oak_camera)
    dexhand_encoder = (
        XhandEncoder("inspire", verbose=False, uart_port="/dev/ttyACM0")
        if hand_type == "xhand"
        else InspireEncoder("inspire", verbose=False, uart_port="/dev/ttyACM0")
    )
    dexhand_encoder.start_streaming()

    if hand_type == "xhand":
        per_finger_adj_val = [0] * 12
        per_finger_adj_val[0] = 4.5
        per_finger_adj_val[1] = -3
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
        print("init")
        time.sleep(2)
        hand.start_reader()
        # Example hand ID and register values
        control_frequency = 30
        while True:
            with FrameRateContext(
                frame_rate=control_frequency, verbose=False
            ) as frame_manager:
                if cameras:
                    frames = [camera.get_camera_frame() for camera in cameras]
                    if len(frames) >= 2:
                        # Ensure frames have same height
                        height = min(
                            int(frames[0].rgb.shape[0] * 0.8),
                            int(frames[1].rgb.shape[0] * 0.8),
                        )
                        width = min(
                            int(frames[0].rgb.shape[1] * 0.8),
                            int(frames[1].rgb.shape[1] * 0.8),
                        )

                        # Resize frames to same dimensions
                        frame1 = cv2.resize(frames[0].rgb, (width, height))
                        frame2 = cv2.resize(frames[1].rgb, (width, height))

                        # Create overlay by blending the frames
                        alpha = 0.5  # Transparency factor
                        overlay_frame = cv2.addWeighted(
                            frame1, alpha, frame2, 1 - alpha, 0
                        )

                        # Concatenate frames horizontally
                        combined_frame = cv2.hconcat([frame1, frame2, overlay_frame])

                        # Resize combined frame to half size
                        display_width = combined_frame.shape[1] // 2
                        display_height = combined_frame.shape[0] // 2
                        combined_frame = cv2.resize(
                            combined_frame, (display_width, display_height)
                        )

                        # Show combined frame
                        cv2.imshow("All Views", combined_frame)

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            for camera in cameras:
                                camera.stop_streaming()
                            dexhand_encoder.stop_streaming()
                            exit()
                        elif key == ord("s"):
                            break

                numeric_frame = dexhand_encoder.get_numeric_frame()
                joint_angles = numeric_frame.joint_angles
                predicted_motor_value = hand.predict_motor_value(joint_angles)
                command = hand.write_hand_angle_position_from_motor(
                    predicted_motor_value
                )
                hand.send_command(command)


# Main function
if __name__ == "__main__":
    overlay()
