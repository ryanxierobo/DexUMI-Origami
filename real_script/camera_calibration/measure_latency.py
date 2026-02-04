import argparse
import time

import cv2
import numpy as np
from dexumi.camera.iphone_camera import IphoneCamera
from dexumi.camera.oak_camera import OakCamera, get_all_oak_cameras
from dexumi.camera.qr_code import generate_qr, read_time_from_qr_code
from dexumi.camera.uvc_camera import UvcCamera


def select_camera(camera_type, device_id):
    """Selects the appropriate camera class based on the user's input."""
    if camera_type == "iphone":
        return IphoneCamera()
    elif camera_type == "uvc":
        return UvcCamera(device_id=device_id, camera_name="UVC Camera")
    elif camera_type == "oak":
        oak_cameras = get_all_oak_cameras()
        return OakCamera(camera_name="OAK Camera", device_id=oak_cameras[1])
    else:
        raise ValueError(f"Unknown camera type: {camera_type}")


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Select a camera type to use.")
    parser.add_argument(
        "camera_type",
        choices=["iphone", "uvc", "oak"],
        help="The type of camera to use (iphone or uvc).",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="The device ID for the UVC camera (default is 0). Only applicable for UVC cameras.",
    )

    # Parse command-line arguments
    args = parser.parse_args()
    # Select the camera, passing the device ID for UVC cameras
    camera = select_camera(args.camera_type, args.device_id)
    camera.start_streaming()

    latency_records = []
    while True:
        monotonic_time = time.monotonic()
        # Encode monotonic time into QR code as a string
        dynamic_data = f"Monotonic Time: {monotonic_time}"
        # Generate the QR code with the monotonic time as data
        qr_img = generate_qr(dynamic_data)
        cv2.imshow("QR Code", qr_img)
        # Capture an image (assuming each camera class has this method)
        frame_data = camera.get_camera_frame()
        if frame_data.rgb is not None:
            capture_time = read_time_from_qr_code(frame_data.rgb)
            # device_capture_time = frame_data.capture_time
            cv2.imshow("RGB", frame_data.rgb)
            # Generate a random image for testing purposes
            # random_img = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
            # cv2.imshow("Random Image", random_img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            capture_time = None

        if capture_time is not None:
            receive_time = frame_data.receive_time
            latency = receive_time - capture_time
            latency_records.append(latency)
            print("receive_time:", receive_time)
            print(f"Latency: {latency:.4f} seconds")
            # print(f"Device capture Latency: {device_capture_time-capture_time:.4f} seconds")

    cv2.destroyAllWindows()
    camera.stop_streaming()

    if latency_records:
        print(
            f"Average latency from {len(latency_records)} frames:",
            sum(latency_records) / len(latency_records),
        )
        print("Stddev:", np.std(latency_records))
        print("Max latency:", max(latency_records))
        print("Min latency:", min(latency_records))
        print("Median latency:", np.median(latency_records))


if __name__ == "__main__":
    main()
