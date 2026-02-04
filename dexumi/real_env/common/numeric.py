import pickle
from typing import Any, List, Literal, Optional

import cv2
import numpy as np
from dexumi.camera.camera import Camera, FrameData

from .base import Request, RequestType, ZMQClientBase, ZMQServerBase


class NumericServer(ZMQServerBase):
    def __init__(
        self,
        numeric: Any,
        pub_address: str = "ipc:///tmp/numeric_stream",
        req_address: str = "ipc:///tmp/numeric_req",
        max_buffer_size: int = 30,
        pub_frequency: int = 60,
        req_frequency: int = 60,
        frames_per_publish: int = 1,
        topic: str = "numeric",
    ):
        super().__init__(
            pub_address=pub_address,
            req_address=req_address,
            max_buffer_size=max_buffer_size,
            pub_frequency=pub_frequency,
            req_frequency=req_frequency,
            frames_per_publish=frames_per_publish,
            topic=topic,
        )
        self.numeric = numeric

    def _get_data(self):
        return self.numeric.get_numeric_frame()


class NumericClient(ZMQClientBase):
    def __init__(
        self,
        pub_address: str = "ipc:///tmp/numeric_stream",
        req_address: str = "ipc:///tmp/numeric_req",
        topic: str = "numeric",
    ):
        super().__init__(pub_address=pub_address, req_address=req_address, topic=topic)

    def receive_frame(self, timeout: Optional[int] = None) -> Optional[dict]:
        """
        Receive frames from the stream. Returns a dictionary containing multiple frames
        if K-frame publishing is enabled.
        """
        return self.receive_data(timeout)
