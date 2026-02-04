from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class NumericFrame:
    capture_time: Optional[float] = None
    receive_time: Optional[float] = None


@dataclass
class JointFrame(NumericFrame):
    joint_angles: Optional[List[float]] = None
    raw_voltage: Optional[List[float]] = None


@dataclass
class FSRFrame(NumericFrame):
    fsr_values: Optional[List[float]] = None


class Numeric:
    def __init__(self, device_name: str, latency: float):
        self.device_name = device_name
        self.latency = latency

        self.is_running = False

    @abstractmethod
    def start_streaming(self):
        pass

    @abstractmethod
    def get_numeric_frame(self):
        pass

    @abstractmethod
    def stop_streaming(self):
        pass
