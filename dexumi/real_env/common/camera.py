import pickle
from typing import Any, List, Literal, Optional, Union

import cv2
import numpy as np
from dexumi.camera.camera import Camera, FrameData

from .base import Request, RequestType, Response, ZMQClientBase, ZMQServerBase


class CameraRequestType(RequestType):
    GET_RECENT_K = "get_recent_k"
    GET_INTRINSICS = "get_intrinsics"


ColorConversion = Literal[None, "BGR2RGB", "RGB2BGR"]


# Update the type hints to include compression options
CompressionType = Literal[None, "jpeg"]


class CameraServer(ZMQServerBase):
    def __init__(
        self,
        camera: Camera,
        pub_address: str = "ipc:///tmp/camera_stream",
        req_address: str = "ipc:///tmp/camera_req",
        max_buffer_size: int = 30,
        pub_frequency: int = 60,
        req_frequency: int = 60,
        resize_ratio: float = 1.0,
        color_conversion: ColorConversion = None,
        frames_per_publish: int = 1,
        topic: str = "camera",
        compression: CompressionType = None,
        compression_quality: int = 90,
    ):
        """
        Initialize camera server with flexible transport and compression support.

        Args:
            camera: Camera instance
            pub_address: Full address for PUB socket
            req_address: Full address for REQ socket
            max_buffer_size: Maximum number of frames to buffer
            pub_frequency: Publishing frequency in Hz
            req_frequency: Request handling frequency in Hz
            resize_ratio: Ratio to resize images (1.0 means no resize)
            color_conversion: Color space conversion
            frames_per_publish: Number of frames to publish at once
            topic: Topic name for publishing
            compression: Type of compression to use (None or 'jpeg')
            compression_quality: Compression quality (0-100, only for JPEG)
        """
        super().__init__(
            pub_address=pub_address,
            req_address=req_address,
            max_buffer_size=max_buffer_size,
            pub_frequency=pub_frequency,
            req_frequency=req_frequency,
            frames_per_publish=frames_per_publish,
            topic=topic,
        )
        self.camera = camera
        self.resize_ratio = resize_ratio
        self.color_conversion = color_conversion
        self.compression = compression
        self.compression_quality = max(
            0, min(100, compression_quality)
        )  # Clamp between 0-100

        self._color_conversion_map = {
            "BGR2RGB": cv2.COLOR_BGR2RGB,
            "RGB2BGR": cv2.COLOR_RGB2BGR,
            None: None,
        }

    def compress_frame(self, frame: np.ndarray) -> Union[np.ndarray, bytes]:
        """Compress the frame using the specified compression method."""
        if self.compression == "jpeg":
            # Encode frame to JPEG format
            success, encoded_frame = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.compression_quality]
            )
            if success:
                return encoded_frame.tobytes()
            else:
                return frame  # Return original frame if compression fails
        return frame

    def process_frame(self, frame: np.ndarray) -> Union[np.ndarray, bytes]:
        """Process the frame by resizing, converting color space, and compressing if specified."""
        if frame is None:
            return None

        # Resize if needed
        if self.resize_ratio != 1.0:
            height, width = frame.shape[:2]
            new_width = int(width * self.resize_ratio)
            new_height = int(height * self.resize_ratio)
            frame = cv2.resize(frame, (new_width, new_height))

        # Convert color space if needed
        if self.color_conversion is not None:
            conversion_code = self._color_conversion_map[self.color_conversion]
            if conversion_code is not None:
                frame = cv2.cvtColor(frame, conversion_code)

        # Apply compression if specified
        if self.compression:
            frame = self.compress_frame(frame)

        return frame

    def _get_data(self) -> Optional[FrameData]:
        """Get frame from camera and process it"""
        frame = self.camera.get_camera_frame()
        if frame is not None:
            frame.rgb = self.process_frame(frame.rgb)
        return frame

    def start(self):
        """Start the camera and base server"""
        self.camera.start_streaming()
        super().start()

    def _process_request(self, request: Request) -> Any:
        """Process camera-specific requests"""
        if request.type == CameraRequestType.GET_RECENT_K:
            k = request.params.get("k", 1)
            return self.data_buffer.read_last(k)
        elif request.type == CameraRequestType.GET_INTRINSICS:
            return self.camera.intrinsic_matrix
        return {"error": "Unknown request type"}

    def stop(self):
        """Stop the camera and base server"""
        super().stop()
        self.camera.stop_streaming()

    def clear_frame_buffer(self):
        """Clear the frame buffer"""
        self.data_buffer.clear()


class CameraClient(ZMQClientBase):
    def __init__(
        self,
        pub_address: str = "ipc:///tmp/camera_stream",
        req_address: str = "ipc:///tmp/camera_req",
        topic: str = "camera",
    ):
        """
        Initialize camera client with flexible transport support.

        Args:
            pub_address: Full address for PUB socket (e.g., "tcp://127.0.0.1:5555" or "ipc:///tmp/camera_stream")
            req_address: Full address for REQ socket (e.g., "tcp://127.0.0.1:5556" or "ipc:///tmp/camera_req")
            topic: Topic name for subscribing
        """
        super().__init__(
            pub_address=pub_address,
            req_address=req_address,
            topic=topic,
        )

    def decompress_frame(
        self, frame_data: Union[bytes, np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Decompress frame data if it's compressed.

        Args:
            frame_data: Compressed frame data as bytes or uncompressed numpy array

        Returns:
            Decompressed frame as numpy array or None if decompression fails
        """
        if isinstance(frame_data, bytes):
            try:
                # Decode JPEG bytes to numpy array
                frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                return cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"Error decompressing frame: {e}")
                return None
        return frame_data

    def process_received_frame(self, frame_data: FrameData) -> Optional[FrameData]:
        """
        Process received frame data, including decompression if needed.

        Args:
            frame_data: FrameData object containing potentially compressed frame

        Returns:
            Processed FrameData object with decompressed frame
        """
        if frame_data is None:
            return None

        # Create a copy to avoid modifying the original
        processed_frame = pickle.loads(pickle.dumps(frame_data))

        # Decompress the RGB frame if it exists
        if processed_frame["data"].rgb is not None:
            processed_frame["data"].rgb = self.decompress_frame(
                processed_frame["data"].rgb
            )

        return processed_frame

    def receive_frame(self, timeout: Optional[int] = None) -> Optional[list]:
        """
        Receive frames from the stream and decompress if needed.
        Returns a list containing multiple frames if K-frame publishing is enabled.

        Args:
            timeout: Optional timeout in milliseconds

        Returns:
            List of processed FrameData objects or None if no data received
        """
        frames = self.receive_data(timeout)
        if frames is None:
            return None

        # Process each frame in the received data
        return [self.process_received_frame(frame) for frame in frames]

    def get_recent_frames(self, k: int, timeout=1000) -> List[dict]:
        """Synchronous request for frames with timeout and decompression."""
        request = Request(type=CameraRequestType.GET_RECENT_K, params={"k": k})
        frames = self.send_request(request, timeout)
        if frames is None:
            return []

        return [self.process_received_frame(frame) for frame in frames]

    def get_recent_frames_async(self, k: int):
        """Send asynchronous request for frames"""
        request = Request(type=CameraRequestType.GET_RECENT_K, params={"k": k})
        self.send_request_async(request)

    def get_intrinsics(self, timeout=1000) -> Optional[np.ndarray]:
        """Synchronous request for intrinsics with timeout"""
        request = Request(type=CameraRequestType.GET_INTRINSICS)
        return self.send_request(request, timeout)

    def get_intrinsics_async(self):
        """Send asynchronous request for intrinsics"""
        request = Request(type=CameraRequestType.GET_INTRINSICS)
        self.send_request_async(request)


# Example usage:
if __name__ == "__main__":
    camera = Camera()  # Your camera instance

    # IPC Example (local only)
    ipc_server = CameraServer(
        camera=camera,
        pub_address="ipc:///tmp/camera_stream",
        req_address="ipc:///tmp/camera_req",
    )

    ipc_client = CameraClient(
        pub_address="ipc:///tmp/camera_stream",
        req_address="ipc:///tmp/camera_req",
    )

    # TCP Example for network communication
    # Server side (binds to all interfaces)
    tcp_server = CameraServer(
        camera=camera,
        pub_address="tcp://*:5555",  # Server binds to all interfaces
        req_address="tcp://*:5556",
    )

    # Client side (connects to specific server IP)
    # Replace SERVER_IP with actual server IP (e.g., "192.168.1.100")
    SERVER_IP = "192.168.1.100"  # Example IP
    tcp_client = CameraClient(
        pub_address=f"tcp://{SERVER_IP}:5555",
        req_address=f"tcp://{SERVER_IP}:5556",
    )

    # Local TCP Example (same machine)
    local_tcp_server = CameraServer(
        camera=camera,
        pub_address="tcp://*:5555",
        req_address="tcp://*:5556",
    )

    local_tcp_client = CameraClient(
        pub_address="tcp://localhost:5555",
        req_address="tcp://localhost:5556",
    )
