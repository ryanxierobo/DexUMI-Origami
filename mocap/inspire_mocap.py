import argparse
import time

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import zarr

# import keyboard
from mocap_util.mocap_node import MocapNode
from mpl_toolkits.mplot3d import Axes3D

from dexumi.common.frame_manager import FrameRateContext
from dexumi.hand_sdk.inspire.hand_api_cls import InspireSDK

# Set image dimensions
image_width = 640
image_height = 480


def generate_random_image(width, height):
    """Generate a random image with the given dimensions."""
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


finger_to_index = {
    "thumb_swing": 0,
    "thumb_proximal": 1,
    "index": 2,
    "middle": 3,
    "ring": 4,
    "little": 5,
}

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="172.24.69.240")
    parser.add_argument("--name", type=str, default="index")
    parser.add_argument("--control_frequency", type=int, default=5)
    parser.add_argument(
        "--hand_id", type=int, default=0x01, help="Hand ID (default: 0x01)"
    )
    parser.add_argument(
        "--grids", type=int, default=48, help="Number of grid points (default: 16)"
    )
    parser.add_argument(
        "--swing_value",
        type=int,
        default=800,
        help="Swing value for thumb (default: 800)",
    )
    args = parser.parse_args()
    if args.name not in finger_to_index:
        raise ValueError(f"Invalid finger name: {args.name}")
    hand = InspireSDK(hand_id=args.hand_id)
    connect = hand.connect()
    buffer = zarr.open("inspire_hand_mocap_1-6-low_residual_0.39", mode="a")

    # Modify buffer group name for thumb_proximal
    if args.name == "thumb_proximal":
        buffer_name = f"{args.name}_{args.swing_value}"
    else:
        buffer_name = args.name
    finger_buffer = buffer.require_group(buffer_name)

    if "thumb" in args.name:
        agent = MocapNode(rigid_body_dict={5: "thumb", 4: "base"}, ip=args.ip)
    else:
        agent = MocapNode(rigid_body_dict={5: "fingertips", 4: "base"}, ip=args.ip)
    time.sleep(1)
    swing_value = args.swing_value
    if connect:
        grids = args.grids
        target_motor_value = np.linspace(0, 1000, grids).astype(np.int32)[::-1]
        all_motor_value = np.ones((6, grids), dtype=np.int32) * 1000
        if args.name == "thumb_proximal":
            all_motor_value[0, :] = swing_value
        finger_index = finger_to_index[args.name]
        all_motor_value[finger_index, :] = target_motor_value

        while True:
            current_episode_index = len(finger_buffer)
            episode_data = finger_buffer.create_group(
                f"episode_{current_episode_index}"
            )
            while True:
                with FrameRateContext(
                    frame_rate=args.control_frequency * 2
                ) as frame_manager:
                    rest_value = [1000] * 6
                    if args.name == "thumb_proximal":
                        rest_value[0] = swing_value
                    with FrameRateContext(frame_rate=30) as frame_manager:
                        random_image = generate_random_image(image_width, image_height)
                        command = hand.write_hand_angle(
                            *rest_value[::-1],
                        )
                        hand.send_command(command)
                        cv2.imshow("Random Image Window", random_image)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            exit()
                        elif key == ord("s"):
                            print("Start recording.")
                            break
            fingertips = []
            inspire_hands = []
            iter = 0
            while True:
                with FrameRateContext(
                    frame_rate=args.control_frequency
                ) as frame_manager:
                    command = hand.write_hand_angle(
                        *all_motor_value[:, iter][::-1],
                    )
                    hand.send_command(command)
                    random_image = generate_random_image(image_width, image_height)
                    cv2.imshow("Random Image Window", random_image)
                    key = cv2.waitKey(1)  # waits for 1 ms
                    if key == ord("s"):
                        print("Break signal received.")
                        break
                    iter += 1
                if "thumb" in args.name:
                    fingertips.append(agent.track_markers["thumb"])
                else:
                    fingertips.append(agent.track_markers["fingertips"])

                inspire_hands.append(agent.track_markers["base"])
                if iter == grids:
                    break
            fingertips = np.array(fingertips)
            inspire_hands = np.array(inspire_hands)
            # Print the shapes of the recorded data arrays
            print(f"fingertips.shape: {fingertips.shape}")
            print(f"inspire_hands.shape: {inspire_hands.shape}")
            print(f"target_motor_value.shape: {target_motor_value.shape}")
            episode_data["fingertips_marker"] = fingertips
            episode_data["inspire_hand_marker"] = inspire_hands
            episode_data["motor_value"] = target_motor_value

            # Plot all episodes recorded so far
            plt.figure(figsize=(15, 5))
            for i in range(current_episode_index + 1):
                episode = finger_buffer[f"episode_{i}"]
                episode_fingertips = episode["fingertips_marker"][:]
                episode_motor = episode["motor_value"][:]

            # Create a 3D plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            # Iterate over all episodes recorded so far
            for i in range(current_episode_index + 1):
                episode = finger_buffer[f"episode_{i}"]
                episode_fingertips = episode["fingertips_marker"][:]

                # Iterate over each marker
                for marker_index in range(episode_fingertips.shape[1]):
                    # Extract the x, y, z coordinates for the current marker
                    x = episode_fingertips[:, marker_index, 0]
                    y = episode_fingertips[:, marker_index, 1]
                    z = episode_fingertips[:, marker_index, 2]

                    # Plot the trajectory for the current marker
                    ax.plot(x, y, z, label=f"Episode {i + 1} Marker {marker_index + 1}")

            # Set labels
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title("Fingertips Trajectory")

            # Show legend
            # ax.legend()

            # Show plot
            plt.show()
