import time
from typing import Optional, Tuple

import cv2
import numpy as np
from dexumi.camera.camera import Camera, FrameData
from dexumi.camera.qr_code import generate_qr


class DebugCamera(Camera):
    def __init__(
        self,
        camera_name: str,
        camera_resolution: Tuple[int, int] = (1280, 800),
        latency: Optional[float] = 0.0,
        fps: int = 60,
    ):
        super().__init__(
            camera_name=camera_name,
            camera_resolution=camera_resolution,
            latency=latency,
        )
        self.fps = fps
        self.is_running = False
        self.last_frame_time = 0
        self.frame_interval = 1.0 / fps

    def create_debug_frame(self):
        # Generate QR code with timestamp
        monotonic_time = time.monotonic()

        # Encode monotonic time into QR code as a string
        dynamic_data = f"Monotonic Time: {monotonic_time}"

        # Generate the QR code with the monotonic time as data
        qr_img = generate_qr(dynamic_data)
        qr_img = cv2.resize(qr_img, (1280, 800))

        return qr_img

    def get_camera_frame(self):
        if self.is_running:
            # Create frame with QR code
            frame = self.create_debug_frame()
            current_time = time.monotonic()
            if self.latency is not None:
                capture_time = current_time - self.latency
            else:
                capture_time = current_time

            return FrameData(
                rgb=frame,
                capture_time=capture_time,
                receive_time=current_time,
            )

    def start_streaming(self):
        self.is_running = True
        print(f"Camera-{self.camera_name} started.")

    def stop_streaming(self):
        self.is_running = False
        print(f"Camera-{self.camera_name} stopped.")
        cv2.destroyAllWindows()
