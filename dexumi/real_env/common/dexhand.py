import os
import time
import traceback
from queue import Empty, Full, Queue
from threading import Lock, Thread
from typing import Any, List, Literal, Optional, Union

import numpy as np

from dexumi.common.frame_manager import FrameRateContext

from .base import Request, RequestType, ZMQClientBase, ZMQServerBase
from .motor_trajectory_interpolator import MotorTrajectoryInterpolator


class DexRequestType(RequestType):
    STOP = "stop"
    SCHEDULE_WAYPOINT = "schedule_waypoint"
    SEND_POS = "send_pos"
    GET_POS = "get_pos"
    GET_TACTILE = "get_tactile"
    PREDICT_POS_FROM_JOINT = "predict_pos_from_joint"


class DexServer(ZMQServerBase):
    def __init__(
        self,
        hand: Any,
        pub_address: str = "ipc:///tmp/dex_stream",
        req_address: str = "ipc:///tmp/dex_req",
        max_buffer_size: int = 30,
        pub_frequency: int = 60,
        req_frequency: int = 60,
        frames_per_publish: int = 1,
        topic: str = "dexhand",
        frequency: int = 20,  # Control loop frequency
        max_motor_speed: float = 1000.0,  # Maximum motor speed
        launch_timeout: float = 3.0,
        soft_real_time: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize InspireServer with trajectory interpolation capabilities.

        Args:
            hand: hand SDK instance
            pub_address: Path for IPC publisher
            req_address: Path for IPC requests
            max_buffer_size: Maximum buffer size
            pub_frequency: Publishing frequency
            req_frequency: Request handling frequency
            frequency: Control loop frequency
            max_motor_speed: Maximum motor movement speed
            launch_timeout: Timeout for initialization
            soft_real_time: Enable real-time scheduling
            verbose: Enable verbose logging
        """
        assert 0 < frequency <= 500
        assert 0 < max_motor_speed

        self.hand = hand
        self.frequency = frequency
        self.max_motor_speed = max_motor_speed
        self.launch_timeout = launch_timeout
        self.soft_real_time = soft_real_time
        self.verbose = verbose

        self.input_queue = Queue(maxsize=100)

        super().__init__(
            pub_address=pub_address,
            req_address=req_address,
            max_buffer_size=max_buffer_size,
            pub_frequency=pub_frequency,
            req_frequency=req_frequency,
            frames_per_publish=frames_per_publish,
            topic=topic,
            verbose=verbose,
        )

    def _get_data(self):
        """Get current state data."""
        # current_pos = self.inspire.get_current_position()
        # return {
        #     "current_position": current_pos,
        # }
        return [0, 0, 0]

    def _process_request(self, request: Request) -> Any:
        """Process incoming requests."""
        if request.type in [
            DexRequestType.STOP,
            DexRequestType.SCHEDULE_WAYPOINT,
            DexRequestType.SEND_POS,
        ]:
            self.input_queue.put(request)
        elif request.type == DexRequestType.GET_POS:
            return self.hand.get_current_position()
        elif request.type == DexRequestType.PREDICT_POS_FROM_JOINT:
            joint_angles = request.params["joint_angles"]
            return self.hand.predict_motor_value(joint_angles)
        elif request.type == DexRequestType.GET_TACTILE:
            calc = request.params.get("calc", False)
            return self.hand.get_tactile(calc=calc)
        else:
            return {"error": "Unknown request type"}

    def start(self):
        """Start server threads and control loop."""
        # Start base server
        super().start()

        # Create and start control loop thread
        self.run_thread = Thread(target=self.run, name="Control-Loop")
        self.run_thread.daemon = False
        self.run_thread.start()

        # Wait for initialization
        time.sleep(self.launch_timeout)

    def run(self):
        """Main control loop."""
        # Enable soft real-time scheduling if requested
        if self.soft_real_time:
            try:
                os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))
            except Exception as e:
                if self.verbose:
                    print(f"Failed to set real-time scheduling: {e}")

        try:
            if self.verbose:
                print("Starting Inspire control loop")

            dt = 1.0 / self.frequency
            curr_pos = self.hand.get_current_position()
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            motor_interp = MotorTrajectoryInterpolator(
                times=[curr_t], values=[curr_pos]
            )

            while self.running:
                with FrameRateContext(self.frequency, verbose=False) as fr:
                    # Get interpolated position command
                    t_now = time.monotonic()
                    pos_command = motor_interp(t_now)
                    # pos_command = np.array(pos_command, dtype=np.int32)
                    # pos_command = np.clip(pos_command, 0, 1000)
                    self._debug(f"target position: {pos_command}")
                    # Send command to device
                    command = self.hand.write_hand_angle_position_from_motor(
                        pos_command
                    )
                    self.hand.send_command(command)

                    # Handle incoming requests
                    try:
                        req = self.input_queue.get(timeout=dt)
                        self._debug(f"Received request: {req}")
                    except Empty:
                        continue

                    if req.type == DexRequestType.STOP:
                        pass
                    elif req.type == DexRequestType.SCHEDULE_WAYPOINT:
                        target_pos = req.params["target_pos"]
                        target_time = float(req.params["target_time"])
                        print("Scheduling waypoint", target_pos, target_time)
                        # Convert global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt

                        # Update trajectory
                        motor_interp = motor_interp.schedule_waypoint(
                            value=target_pos,
                            time=target_time,
                            max_speed=self.max_motor_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time,
                        )
                        last_waypoint_time = target_time
                    elif req.type == DexRequestType.SEND_POS:
                        # Immediate position command
                        pos = req.params["pos"]
                        print("Immediate position command", pos)
                        command = self.hand.write_hand_angle_position_from_motor(pos)
                        self.hand.send_command(command)

                    if self.verbose:
                        print(
                            f"Control loop frequency: {1 / (time.monotonic() - t_now)}"
                        )

        except Exception as e:
            if self.verbose:
                print(f"Error in control loop: {e}")
                print(traceback.format_exc())

    def stop(self):
        """Stop server and cleanup."""
        if not hasattr(self, "_stopped") or not self._stopped:
            try:
                # Stop control loop
                self.input_queue.put(Request(type=DexRequestType.STOP))

                # Wait for run thread
                if hasattr(self, "run_thread") and self.run_thread.is_alive():
                    self.run_thread.join(timeout=5.0)

                # Stop base server
                super().stop()

                # Cleanup
                self.hand.stop_reader()

                self._stopped = True
            except Exception as e:
                print(f"Error during stop: {e}")


class DexClient(ZMQClientBase):
    def __init__(
        self,
        pub_address: str = "ipc:///tmp/dex_stream",
        req_address: str = "ipc:///tmp/dex_req",
        topic: str = "dexhand",
        verbose: bool = False,
    ):
        super().__init__(
            pub_address=pub_address,
            req_address=req_address,
            topic=topic,
            verbose=verbose,
        )

    def schedule_waypoint(
        self,
        target_pos: List[float],
        target_time: float,
        timeout: Optional[float] = None,
    ) -> dict:
        """Schedule a waypoint with trajectory interpolation."""
        req = Request(
            type=DexRequestType.SCHEDULE_WAYPOINT,
            params={"target_pos": target_pos, "target_time": target_time},
        )
        return self.send_request(req, timeout)

    def send_pos(self, pos: List[float], timeout: Optional[float] = None) -> dict:
        """Send immediate position command."""
        if not isinstance(pos, (list, np.ndarray)):
            raise ValueError(f"Expected pos to be list or ndarray, got {type(pos)}")
        req = Request(type=DexRequestType.SEND_POS, params={"pos": pos})
        return self.send_request(req, timeout=timeout)

    def get_pos(self, timeout: Optional[float] = None) -> dict:
        """Get current position."""
        req = Request(type=DexRequestType.GET_POS)
        return self.send_request(req, timeout=timeout)

    def get_tactile(self, calc: bool = False, timeout: Optional[float] = None) -> dict:
        """Get tactile data."""
        req = Request(type=DexRequestType.GET_TACTILE, params={"calc": calc})
        return self.send_request(req, timeout=timeout)

    def predict_pos_from_joint(
        self, joint_angles: List[float], timeout: Optional[float] = None
    ) -> dict:
        """Predict motor positions from joint angles."""
        req = Request(
            type=DexRequestType.PREDICT_POS_FROM_JOINT,
            params={"joint_angles": joint_angles},
        )
        return self.send_request(req, timeout=timeout)

    def get_state(self, timeout: Optional[float] = None) -> dict:
        """Get current state."""
        try:
            req = Request(type=DexRequestType.GET_STATE)
            state = self.send_request(req, timeout)
            if state is None:
                raise TimeoutError("Request timed out")
            return state
        except Exception as e:
            raise RuntimeError(f"Failed to get state: {e}")
