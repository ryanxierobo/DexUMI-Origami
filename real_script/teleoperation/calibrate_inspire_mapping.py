import os  # Add this import
import pickle
import time

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from dexumi.camera.oak_camera import OakCamera, get_all_oak_cameras
from dexumi.common.frame_manager import FrameRateContext
from dexumi.encoder.encoder import InspireEncoder
from dexumi.hand_sdk.inspire.hand_api_cls import InspireSDK

# Simpler relative path from script location to model directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.abspath(
    os.path.join(SCRIPT_DIR, "../../../data_local/visual_regression")
)


@click.command()
@click.option("--enable_encoder", is_flag=True, help="Enable encoder")
@click.option("--joint_index", type=int, default=0, help="Joint index to map (0-5)")
@click.option(
    "--model_dir", type=str, default=DEFAULT_MODEL_DIR, help="Directory to save models"
)
def main(enable_encoder, joint_index, model_dir):
    if not 0 <= joint_index <= 5:
        raise ValueError("Joint index must be between 0 and 5")

    os.makedirs(model_dir, exist_ok=True)

    connected_oak_cameras = get_all_oak_cameras()
    cameras = []
    for camera in connected_oak_cameras:
        oak_camera = OakCamera("oak", device_id=camera)
        oak_camera.start_streaming()
        cameras.append(oak_camera)

    dexhand_encoder = InspireEncoder("inspire", verbose=False)
    dexhand_encoder.start_streaming()
    time.sleep(2)
    hand = InspireSDK(
        port="/dev/ttyUSB0",
        read_rate=30,
    )

    if hand.connect():
        print(f"Initializing mapping for joint index {joint_index}")

        control_frequency = 30
        dt = 1 / control_frequency

        # Generate target motor values for the specified joint
        if joint_index == 0:
            target_motor_value = np.linspace(0, 1000, 12).astype(np.int32)
        else:
            target_motor_value = np.linspace(0, 1000, 12).astype(np.int32)[::-1]

        iter = 0
        encoder_values = []
        exec = True

        while True:
            with FrameRateContext(
                frame_rate=control_frequency, verbose=False
            ) as frame_manager:
                numeric_frame = dexhand_encoder.get_numeric_frame()
                joint_angles = numeric_frame.joint_angles
                joint_angles = np.array(joint_angles).astype(np.int32)

                if cameras:
                    frames = [camera.get_camera_frame() for camera in cameras]
                    if len(frames) >= 2:
                        # Ensure frames have same height
                        height = min(frames[0].rgb.shape[0], frames[1].rgb.shape[0])
                        width = min(frames[0].rgb.shape[1], frames[1].rgb.shape[1])

                        frame1 = cv2.resize(frames[0].rgb, (width, height))
                        frame2 = cv2.resize(frames[1].rgb, (width, height))

                        alpha = 0.5
                        overlay_frame = cv2.addWeighted(
                            frame1, alpha, frame2, 1 - alpha, 0
                        )

                        combined_frame = cv2.hconcat([frame1, frame2, overlay_frame])

                        display_width = combined_frame.shape[1] // 2
                        display_height = combined_frame.shape[0] // 2
                        combined_frame = cv2.resize(
                            combined_frame, (display_width, display_height)
                        )

                        cv2.imshow("All Views", combined_frame)

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            for camera in cameras:
                                camera.stop_streaming()
                            dexhand_encoder.stop_streaming()
                            exit()
                        elif key == ord("s"):
                            if iter < len(encoder_values):
                                exec = True
                                iter += 1
                        elif key == ord("r"):
                            print("record!")
                            if enable_encoder:
                                # Record voltage for the specified joint index
                                encoder_values.append(joint_angles[joint_index])
                            else:
                                encoder_values.append(0)
                            print(encoder_values)

                if iter >= len(target_motor_value):
                    break

                if exec:
                    # Set all joints to 1000 except the target joint
                    desired_joint_angles = np.ones(6) * 1000
                    if joint_index == 1:
                        desired_joint_angles[0] = 0
                    desired_joint_angles[joint_index] = target_motor_value[iter]
                    desired_joint_angles = desired_joint_angles.astype(np.int32)
                    print("desired_joint_angles: ", desired_joint_angles[joint_index])
                    command = hand.write_hand_angle(
                        desired_joint_angles,
                    )
                    hand.send_command(command)
                    exec = False

        print("Done")
        encoder_values = np.array(encoder_values)
        print(
            f"Encoder values shape: {encoder_values.shape}, Target values shape: {target_motor_value.shape}"
        )

        # Fit polynomial regression
        polynomial_degree = 3
        model = make_pipeline(
            PolynomialFeatures(degree=polynomial_degree, include_bias=False),
            LinearRegression(),
        )

        X = encoder_values.reshape(-1, 1)
        model.fit(X, target_motor_value)

        # Save model with joint index in filename
        model_filename = os.path.join(
            model_dir, f"joint_to_motor_index_{joint_index}.pkl"
        )
        with open(model_filename, "wb") as f:
            pickle.dump(model, f)

        # Make predictions for visualization
        predictions = model.predict(X)

        # Plot results
        plt.figure(figsize=(10, 6))
        plt.scatter(
            encoder_values, target_motor_value, color="blue", label="Actual Values"
        )
        plt.plot(encoder_values, predictions, color="red", label="Predicted Values")
        plt.xlabel("Encoder Values")
        plt.ylabel("Motor Values")
        plt.title(
            f"Motor Value Prediction for Joint {joint_index} (Polynomial Degree={polynomial_degree})"
        )
        plt.legend()
        plt.grid(True)
        plt.show()

        hand.disconnect()


if __name__ == "__main__":
    main()
