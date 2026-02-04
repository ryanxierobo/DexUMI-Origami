import os
import pickle
import time
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Lock, Thread
from typing import Any, List, Literal, Optional, Union

import numpy as np
from dexumi.common.frame_manager import FrameRateContext
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from .base import Request, RequestType, ZMQClientBase, ZMQServerBase
from .pose_trajectory_interpolator import PoseTrajectoryInterpolator


class UR5RequestType(RequestType):
    START = "start"
    STOP = "stop"
    GET_STATE = "get_state"
    GET_STATE_HISTORY = "get_state_history"
    SCHEDULE_WAYPOINT = "schedule_waypoint"
    SERVOL = "servol"


@dataclass
class UR5Frame:
    capture_time: Optional[float] = None
    receive_time: Optional[float] = None
    state: Optional[dict] = None


class UR5Server(ZMQServerBase):
    def __init__(
        self,
        robot_ip: str,
        pub_address: str = "ipc:///tmp/ur5_stream",
        req_address: str = "ipc:///tmp/ur5_req",
        max_buffer_size: int = 30,
        pub_frequency: int = 60,
        req_frequency: int = 60,
        frames_per_publish: int = 1,
        topic: str = "ur5",
        frequency=500,
        lookahead_time=0.1,
        gain=300,
        max_pos_speed=0.25,  # 5% of max speed
        max_rot_speed=0.16,  # 5% of max speed
        launch_timeout=3,
        tcp_offset_pose=None,
        payload_mass=None,
        payload_cog=None,
        joints_init=None,
        joints_init_speed=1.05,
        soft_real_time=False,
        verbose=False,
        receive_keys=None,
        receive_latency=0.0,
    ):
        """
        Initialize UR5Server with configurable observation requirements.

        Args:
            pub_address (str): Path for IPC publisher
            req_address (str): Path for IPC requests
            max_buffer_size (int): Maximum buffer size
            pub_frequency (int): Publishing frequency
            req_frequency (int): Request handling frequency
            frequency: CB2=125, UR3e=500
            lookahead_time: [0.03, 0.2]s smoothens the trajectory with this lookahead time
            gain: [100, 2000] proportional gain for following target position
            max_pos_speed: m/s
            max_rot_speed: rad/s
            tcp_offset_pose: 6d pose
            payload_mass: float
            payload_cog: 3d position, center of gravity
            soft_real_time: enables round-robin scheduling and real-time priority
                requires running scripts/rtprio_setup.sh before hand.
        """
        assert 0 < frequency <= 500
        assert 0.03 <= lookahead_time <= 0.2
        assert 100 <= gain <= 2000
        assert 0 < max_pos_speed
        assert 0 < max_rot_speed
        if tcp_offset_pose is not None:
            tcp_offset_pose = np.array(tcp_offset_pose)
            assert tcp_offset_pose.shape == (6,)
        if payload_mass is not None:
            assert 0 <= payload_mass <= 5
        if payload_cog is not None:
            payload_cog = np.array(payload_cog)
            assert payload_cog.shape == (3,)
            assert payload_mass is not None
        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (6,)

        self.robot_ip = robot_ip
        self.frequency = frequency
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.tcp_offset_pose = tcp_offset_pose
        self.payload_mass = payload_mass
        self.payload_cog = payload_cog
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.receive_latency = receive_latency
        self.verbose = verbose

        self.input_queue = Queue(maxsize=100)
        self.rtde_r = RTDEReceiveInterface(hostname=robot_ip)
        self.rtde_c = RTDEControlInterface(hostname=robot_ip)
        # build ring buffer
        if receive_keys is None:
            receive_keys = [
                "ActualTCPPose",
                "ActualTCPSpeed",
                "ActualQ",
                "ActualQd",
                "TargetTCPPose",
                "TargetTCPSpeed",
                "TargetQ",
                "TargetQd",
            ]
        self.receive_keys = receive_keys

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

    def _process_request(self, request: Request) -> Any:
        """Process requests for UR5."""
        if request.type == UR5RequestType.GET_STATE:
            self._debug("Received GET_STATE request")
            return self._get_data()
        elif request.type == UR5RequestType.GET_STATE_HISTORY:
            return self.get_state_history()
        elif (
            request.type == UR5RequestType.STOP
            or request.type == UR5RequestType.SCHEDULE_WAYPOINT
        ):
            self.input_queue.put(request)
        elif request.type == UR5RequestType.START:
            self._debug("Received START request")
            self.start()
        else:
            return {"error": "Unknown request type"}

    def _get_data(self):
        """Get data from UR5."""
        state = dict()
        for key in self.receive_keys:
            state[key] = np.array(getattr(self.rtde_r, "get" + key)())
        t_now = time.monotonic()
        return UR5Frame(capture_time=t_now, receive_time=t_now, state=state)

    def get_state_history(self):
        frames = self.data_buffer.read_last(self.frames_per_publish)
        return frames

    def start(self):
        """Start the server threads including the run loop thread"""
        # First start the base server (pub/sub threads)
        super().start()

        # Create and start the run thread
        self.run_thread = Thread(target=self.run, name="UR5-Control-Loop")
        self.run_thread.daemon = False  # Keep running even if main thread exits
        self.run_thread.start()

        # Wait for initialization
        time.sleep(self.launch_timeout)

    def run(self):
        """Main control loop running in background thread"""
        # enable soft real-time
        if self.soft_real_time:
            try:
                os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))
            except Exception as e:
                if self.verbose:
                    print(f"Failed to set real-time scheduling: {e}")

        try:
            if self.verbose:
                print(f"[RTDEPositionalController] Connect to robot: {self.robot_ip}")

            # set parameters
            if self.tcp_offset_pose is not None:
                self.rtde_c.setTcp(self.tcp_offset_pose)
            if self.payload_mass is not None:
                if self.payload_cog is not None:
                    assert self.rtde_c.setPayload(self.payload_mass, self.payload_cog)
                else:
                    assert self.rtde_c.setPayload(self.payload_mass)

            # init pose
            if self.joints_init is not None:
                assert self.rtde_c.moveJ(self.joints_init, self.joints_init_speed, 1.4)

            # main loop
            dt = 1.0 / self.frequency
            curr_pose = self.rtde_r.getActualTCPPose()
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(times=[curr_t], poses=[curr_pose])

            while self.running:  # Use flag from base class
                # start control iteration
                with FrameRateContext(self.frequency, verbose=False) as fr:
                    t_now = time.monotonic()
                    # send command to robot
                    pose_command = pose_interp(t_now)
                    vel = 0.5
                    acc = 0.5
                    assert self.rtde_c.servoL(
                        pose_command,
                        vel,
                        acc,  # dummy, not used by ur5
                        dt,
                        self.lookahead_time,
                        self.gain,
                    )

                    # fetch command from queue with timeout
                    try:
                        req = self.input_queue.get(timeout=dt)
                    except Empty:
                        continue

                    if req.type == UR5RequestType.STOP:
                        # break  # Exit loop cleanly
                        pass
                    elif req.type == UR5RequestType.SERVOL:
                        pass  # Implement if needed
                    elif req.type == UR5RequestType.SCHEDULE_WAYPOINT:
                        target_pose = req.params.get("target_pose")
                        target_time = float(req.params.get("target_time"))
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time,
                        )
                        last_waypoint_time = target_time

                    if self.verbose:
                        print(
                            f"[RTDEPositionalController] Actual frequency {1 / (time.monotonic() - t_now)}"
                        )

        except Exception as e:
            if self.verbose:
                print(f"Error in control loop: {e}")
        # finally:
        #     # mandatory cleanup
        #     try:
        #         self.rtde_c.servoStop()
        #         self.rtde_c.stopScript()
        #         self.rtde_c.disconnect()
        #         self.rtde_r.disconnect()

        #         if self.verbose:
        #             print(
        #                 f"[RTDEPositionalController] Disconnected from robot: {self.robot_ip}"
        #             )
        #     except Exception as e:
        #         if self.verbose:
        #             print(f"Error during cleanup: {e}")

    def stop(self):
        """Stop all threads and clean up"""
        if not hasattr(self, "_stopped") or not self._stopped:
            try:
                # First stop the control loop
                self.input_queue.put(Request(type=UR5RequestType.STOP))

                # Wait for run thread to finish with timeout
                if hasattr(self, "run_thread") and self.run_thread.is_alive():
                    self.run_thread.join(timeout=5.0)  # 5 second timeout

                # Stop base server threads
                super().stop()

                # Cleanup robot connection
                self.rtde_c.servoStop()
                self.rtde_c.stopScript()
                self.rtde_c.disconnect()
                self.rtde_r.disconnect()

                self._stopped = True
            except Exception as e:
                print(f"Error during stop: {e}")


class UR5eClient(ZMQClientBase):
    def __init__(
        self,
        pub_address: str = "ipc:///tmp/ur5_stream",
        req_address: str = "ipc:///tmp/ur5_req",
        topic: str = "ur5",
        verbose: bool = False,
    ):
        super().__init__(
            pub_address=pub_address,
            req_address=req_address,
            topic=topic,
            verbose=verbose,
        )
        # self.connect_to_robot()

    def schedule_waypoint(
        self, target_pose, target_time, timeout: Optional[int] = None
    ):
        req = Request(
            type=UR5RequestType.SCHEDULE_WAYPOINT,
            params={"target_pose": target_pose, "target_time": target_time},
        )
        self.send_request(req, timeout)

    def stop(self, timeout: Optional[int] = None):
        req = Request(type=UR5RequestType.STOP)
        self.send_request(req, timeout)

    def get_state(self, timeout: Optional[int] = None):
        try:
            req = Request(type=UR5RequestType.GET_STATE)
            state = self.send_request(req, timeout)
            if state is None:
                raise TimeoutError("Request timed out")
            return state
        except Exception as e:
            raise RuntimeError(f"Failed to get state: {e}")

    def get_state_history(self, timeout: Optional[int] = None):
        req = Request(type=UR5RequestType.GET_STATE_HISTORY)
        state_history = self.send_request(req, timeout)
        return state_history

    def connect_to_robot(self, timeout: Optional[int] = None):
        req = Request(type=UR5RequestType.START)
        self.send_request(req, timeout)
