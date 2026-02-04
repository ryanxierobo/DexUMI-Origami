import os
import time

import click
import cv2
import numpy as np
import zarr

from dexumi.camera.camera import FrameData, FrameReplayData
from dexumi.camera.iphone_camera import IphoneCamera
from dexumi.camera.oak_camera import OakCamera, get_all_oak_cameras
from dexumi.common.frame_manager import FrameRateContext
from dexumi.common.utility.file import read_pickle
from dexumi.data_recording import NumericRecorder, VideoRecorder
from dexumi.data_recording.record_manager import RecorderManager
from dexumi.encoder.encoder import InspireEncoder, JointFrame, XhandEncoder
from dexumi.encoder.fsr import FSRSensor
from dexumi.encoder.xhand_tactile import XhandTactile

# Replace the single threshold with a tuple for each index
VOLTAGE_THRESHOLDS = (
    10,
    10,
    10,
    10,
    10,
    10,
    10,
    10,
    10,
    10,
    10,
    10,
)  # Adjust individual thresholds as needed

FINGER_LIMIT = [
    300,
    196,
    233.91633614419237,
    237.2309326180711,
    242.0509145360444,
    241.9494664793761,
    300,
    196,
    233.91633614419237,
    237.2309326180711,
    242.0509145360444,
    241.9494664793761,
]


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(),
    default="~/Dev/DexUMI/data_local/data_recording",
    help="Base directory for storing recorded data",
)
@click.option(
    "--reference-dir",
    type=click.Path(),
    default="~/Dev/DexUMI/data_local/reference_data",
    help="Base directory for reference data",
)
@click.option("--fps", type=int, default=60, help="Recording frame rate")
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
@click.option("--enable_tracking", "-et", is_flag=True, help="Enable verbose logging")
@click.option("--enable_fsr", "-ef", is_flag=True, help="Enable verbose logging")
@click.option("--is_replay", "-r", is_flag=True, help="further record depth data")
@click.option("--hand_type", type=str, default="xhand", help="xhand/inspire")
def record_exoskeleton(
    data_dir: str,
    reference_dir: str,
    fps: int,
    verbose: bool,
    enable_tracking: bool,
    enable_fsr: bool,
    is_replay: bool,
    hand_type: str,
):
    # """Record synchronized video and numeric data from dexumi."""
    # Configure paths
    print("enable_tracking", enable_tracking)
    assert hand_type in ["xhand", "inspire"]
    data_dir = os.path.expanduser(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    try:
        reference_dir = os.path.expanduser(reference_dir)
        reference_buffer = zarr.open(
            os.path.join(reference_dir, "reference_episode"), mode="r"
        )
        reference_encoder_values = reference_buffer["numeric_0"]["joint_angles"][0]
        reference_thumb_data = read_pickle(
            file_name=os.path.join(reference_dir, "exo_thumb_points.pkl")
        )
        reference_thumb_points = [
            point
            for point, label in zip(
                reference_thumb_data["points"], reference_thumb_data["labels"]
            )
            if label == 1
        ]

        reference_finger_data = read_pickle(
            file_name=os.path.join(reference_dir, "exo_finger_points.pkl")
        )
        reference_finger_points = [
            point
            for point, label in zip(
                reference_finger_data["points"], reference_finger_data["labels"]
            )
            if label == 1
        ]
        print(reference_encoder_values)
    except Exception as e:
        print(e)
        click.echo("Error: Failed to load reference data", err=True)
        reference_encoder_values = None
        reference_thumb_points = None
        reference_finger_points = None
        raise e

    # Initialize data sources
    oak_cameras = get_all_oak_cameras()
    if not oak_cameras:
        click.echo("Error: No OAK cameras found", err=True)
        return
    if enable_tracking:
        camera_sources = [
            OakCamera("oak", fps=fps, device_id=oak_cameras[0]),
            IphoneCamera(),
        ]
    else:
        camera_sources = [OakCamera("oak", fps=fps, device_id=oak_cameras[0])]

    encoder_type = InspireEncoder if hand_type == "inspire" else XhandEncoder
    fsr_type = FSRSensor if hand_type == "inspire" else XhandTactile
    numeric_sources = [
        encoder_type(hand_type, uart_port="/dev/ttyACM0", verbose=verbose)
    ]
    if enable_fsr:
        fsr_name = (
            ["xhand_thumbs", "xhand_index", "xhand_middle"]
            if hand_type == "xhand"
            else ["inspire_fsr"]
        )
        fsr_port = (
            ["/dev/ttyACM3", "/dev/ttyACM2", "/dev/ttyACM1"]
            if hand_type == "xhand"
            else ["/dev/ttyACM1"]
        )
        numeric_sources.extend(
            [
                fsr_type(device_name=name, uart_port=port, verbose=verbose)
                for name, port in zip(fsr_name, fsr_port)
            ]
        )

    # Initialize recorders
    video_recorder = VideoRecorder(
        record_fps=fps,
        stream_fps=60,
        video_record_path=data_dir,
        camera_sources=camera_sources,
        frame_data_class=FrameReplayData if is_replay else FrameData,
        # verbose=verbose,
        verbose=True,
    )

    numeric_recorder = NumericRecorder(
        record_fps=fps,
        stream_fps=fps,
        record_path=data_dir,
        numeric_sources=numeric_sources,
        frame_data_class=JointFrame,
        verbose=verbose,
    )

    # Create recorder manager
    recorder_manager = RecorderManager(
        recorders=[video_recorder, numeric_recorder], verbose=verbose
    )

    # Start streaming
    if not recorder_manager.start_streaming():
        click.echo("Failed to start streaming", err=True)
        return

    # if not viz_manager.start_streaming():
    #     click.echo("Failed to start streaming", err=True)
    #     return

    click.echo("\nControls:")
    click.echo("s - Start recording")
    click.echo("w - Stop and save recording")
    click.echo("a - Restart episode")
    click.echo("q - Quit")

    def draw_points(frame, points, color):
        """Draw points on frame with given color."""
        for point in points:
            x, y = int(point[0]), int(point[1])
            cv2.circle(frame, (x, y), 3, color, -1)
            cv2.circle(frame, (x, y), 5, color, 1)

    try:
        start_time = time.monotonic()
        frame_count = 0
        while True:
            with FrameRateContext(fps, verbose=verbose) as fr:
                # Get latest frames from all recorders
                frames = recorder_manager.get_latest_frames()
                try:
                    video_frame = frames["oak"][-1]
                except Exception as e:
                    video_frame = None
                if enable_tracking:
                    tracking_frame = frames["IphoneCamera"][-1]
                    if tracking_frame is None:
                        raise ValueError("Tracking frame is None")
                else:
                    tracking_frame = None

                if enable_fsr:
                    force_frames = [frames[sfn][-1] for sfn in fsr_name]
                    if force_frames is None:
                        raise ValueError("Force frame is None")
                else:
                    force_frames = None

                try:
                    numeric_frame = frames[hand_type][-1]
                except Exception as e:
                    numeric_frame = None

                if video_frame is not None:
                    # Visualize frame
                    viz_frame = video_frame.rgb.copy()
                    current_time = time.monotonic()
                    frame_count += 1
                    print(
                        "viz fps",
                        frame_count / (current_time - start_time),
                    )
                    # Draw reference points
                    if (
                        not recorder_manager.is_recording
                        and reference_thumb_points is not None
                    ):
                        draw_points(
                            viz_frame, reference_thumb_points, (0, 255, 0)
                        )  # Green for thumb
                        draw_points(
                            viz_frame, reference_finger_points, (0, 0, 255)
                        )  # Red for finger

                    cv2.putText(
                        viz_frame,
                        f"Episode: {recorder_manager.episode_id}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                    # Add numeric data overlay if available
                    if numeric_frame is not None:
                        # Initialize as green
                        text_color = (0, 255, 0)

                        if (
                            not recorder_manager.is_recording
                            and reference_encoder_values is not None
                        ):
                            voltage_diffs = abs(
                                numeric_frame.joint_angles - reference_encoder_values
                            )

                            # Modify the voltage difference check in the main loop
                            for i, (voltage, diff) in enumerate(
                                zip(numeric_frame.joint_angles, voltage_diffs)
                            ):
                                if (
                                    diff > VOLTAGE_THRESHOLDS[i]
                                ):  # Use indexed threshold
                                    text_color = (0, 0, 255)  # Red for this value
                                else:
                                    text_color = (0, 255, 0)  # Green for this value

                                # Add each angle with its own color
                                cv2.putText(
                                    viz_frame,
                                    f"Angle {i}: {voltage:.2f}",
                                    (10, 60 + i * 30),  # Stack vertically
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    text_color,
                                    2,
                                )
                        else:
                            # During recording, show all values in green
                            for i, voltage in enumerate(numeric_frame.joint_angles):
                                # Set color to red if voltage is below FINGER_LIMIT
                                text_color = (
                                    (0, 0, 255)
                                    if voltage > FINGER_LIMIT[i]
                                    else (0, 255, 0)
                                )

                                cv2.putText(
                                    viz_frame,
                                    f"Angle {i}: {voltage:.2f}",
                                    (10, 60 + i * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    text_color,  # Red if below threshold, green otherwise
                                    2,
                                )
                            pass

                    if tracking_frame is not None:
                        # Add pose information from tracking frame
                        cv2.putText(
                            viz_frame,
                            f"Pose: {tracking_frame.pose[:3, 3]}",
                            (10, 250),  # Position below other text
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 0),  # Yellow color
                            2,
                        )
                        # Show tracking frame RGB
                        cv2.imshow("Tracking Camera", tracking_frame.rgb)

                    if force_frames is not None:
                        # Add force sensor information for each frame
                        y_offset = 430
                        for frame_idx, force_frame in enumerate(force_frames):
                            if force_frame is not None:
                                cv2.putText(
                                    viz_frame,
                                    f"=== {fsr_name[frame_idx]} ===",
                                    (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (255, 165, 0),
                                    2,
                                )
                                y_offset += 30
                                for i, force in enumerate(force_frame.fsr_values):
                                    cv2.putText(
                                        viz_frame,
                                        f"Force {i}: {force:.2f}",
                                        (10, y_offset + i * 30),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7,
                                        (255, 165, 0),  # Orange color
                                        2,
                                    )
                                y_offset += len(force_frame.fsr_values) * 30 + 10

                    cv2.imshow("Recording Preview", viz_frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        if recorder_manager.stop_recording():
                            recorder_manager.clear_recording()
                        recorder_manager.stop_streaming()
                        break
                    elif key == ord("s"):
                        start_time = time.monotonic()
                        frame_count = 0
                        if recorder_manager.reset_episode_recording():
                            click.echo("Starting recording...")
                            recorder_manager.start_recording()
                        else:
                            click.echo("Recording already started.")
                    elif key == ord("w"):
                        click.echo("Saving recording...")
                        if recorder_manager.stop_recording():
                            recorder_manager.save_recordings()
                    elif key == ord("a"):
                        start_time = time.monotonic()
                        frame_count = 0
                        click.echo("Restarting episode...")
                        if recorder_manager.stop_recording():
                            recorder_manager.clear_recording()

    finally:
        recorder_manager.stop_recording()
        recorder_manager.stop_streaming()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    record_exoskeleton()
