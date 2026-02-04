import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from xhand_controller import xhand_control

from dexumi.hand_sdk.dexhand import DexterousHand, ExoDexterousHand

script_dir = os.path.dirname(os.path.realpath(__file__))
xhandcontrol_library_dir = os.path.join(script_dir, "lib")
os.environ["LD_LIBRARY_PATH"] = (
    xhandcontrol_library_dir + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
)
print(f"LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}\n")


@dataclass
class JointState:
    id: int
    position: float
    raw_position: int
    sensor_id: int
    temperature: int
    torque: int
    commboard_err: int
    jonitboard_err: int  # Note: keeping the typo from original data
    tipboard_err: int
    default5: int
    default6: int
    default7: int


@dataclass
class Force:
    fx: float
    fy: float
    fz: float


@dataclass
class FingertipState:
    calc_pressure: List[float]
    raw_pressure: List[List[float]]
    sensor_temperature: float


@dataclass
class HandState:
    joints: List[JointState]
    fingertips: List[FingertipState]
    error_code: int = 0
    error_message: str = ""


class XhandSDK(DexterousHand):
    def __init__(
        self,
        hand_id=0,
        port="/dev/ttyUSB0",
        protocol="RS485",
        state_queue_size=10,
        update_frequency=30,  # Hz - how many times per second to update position
    ):
        self._device = xhand_control.XHandControl()
        self.port = port
        self.protocol = protocol
        self.running = False
        self._hand_id = hand_id
        self._state_queue = deque(maxlen=state_queue_size)
        self._reader_thread = None
        self._update_frequency = update_frequency
        self._lock = threading.Lock()

    def connect(self):
        device_identifier = {}
        device_identifier["protocol"] = self.protocol
        device_identifier["serial_port"] = self.port
        device_identifier["baud_rate"] = 3000000
        self.open_device(device_identifier)
        self.list_hands_id()
        return True

    def disconnect(self):
        self.stop_reader()
        print("disconnect")

    def open_device(self, device_identifier: dict):
        # RS485
        if device_identifier["protocol"] == "RS485":
            device_identifier["baud_rate"] = int(device_identifier["baud_rate"])
            rsp = self._device.open_serial(
                device_identifier["serial_port"],
                device_identifier["baud_rate"],
            )
            print(f"open RS485 result: {rsp.error_code == 0}\n")
        # EtherCAT
        elif device_identifier["protocol"] == "EtherCAT":
            ether_cat = self.enumerate_devices("EtherCAT")
            print(f"enumerate_devices_ethercat ether_cat= {ether_cat}\n")
            if ether_cat is None or not ether_cat:
                print("enumerate_devices_ethercat get empty \n")
            rsp = self._device.open_ethercat(ether_cat[0])
            print(f"open EtherCAT result: {rsp.error_code == 0}\n")

    def reset_sensor(self, sensor_id):
        print(
            f"xhand reset_sensor result: {self._device.reset_sensor(self._hand_id, sensor_id).error_code == 0}\n"
        )

    def enumerate_devices(self, protocol: str):
        serial_port = self._device.enumerate_devices(protocol)
        print(f"xhand devices port: {serial_port}\n")
        return serial_port

    def list_hands_id(self):
        self._hand_id = self._device.list_hands_id()[0]
        print("-------------------")
        print(f"hand_id: {self._hand_id}\n")
        print("-------------------")

    def set_hand_id(self, new_id):
        hands_id = self._device.list_hands_id()
        print(f"set hand_id before:{hands_id[0]}\n")
        old_id = hands_id[0]
        err_struct = self._device.set_hand_id(old_id, new_id)
        if err_struct.error_code == 0:
            self._hand_id = new_id
        hands_id = self._device.list_hands_id()
        print(f"set hand_id after:{hands_id[0]}\n")
        print(f"xhand set_hand_id result: {err_struct.error_code == 0}\n")

    def get_current_position(self) -> Optional[HandState]:
        """
        Returns the latest hand position from the queue.
        If the queue is empty, attempts to get a position directly.
        """
        with self._lock:
            if not self._state_queue:
                # If queue is empty, try to get position directly
                print("Warning: queue is empty!")
                raise ValueError("Queue is empty")
            state = self._state_queue[-1]  # Return the most recent position
            joint_state = state.joints
            joint_position = []
            for i in range(12):
                joint_position.append(joint_state[i].position)

        return np.array(joint_position)

    def get_tactile(self, calc=False):
        with self._lock:
            if not self._state_queue:
                # If queue is empty, try to get position directly
                print("Warning: queue is empty!")
                raise ValueError("Queue is empty")
            state = self._state_queue[-1]  # Return the most recent position
            fingertips_state = state.fingertips
            fingertips_pressure = []
            for i in range(5):
                if calc:
                    fingertips_pressure.append(fingertips_state[i].calc_pressure)
                else:
                    fingertips_pressure.append(fingertips_state[i].raw_pressure)

        return np.array(fingertips_pressure)

    def _get_current_state(self) -> Optional[HandState]:
        error_struct, state = self._device.read_state(self._hand_id, True)
        if error_struct.error_code != 0:
            print(f"xhand read_state error:{self.parse_error_code(error_struct)}\n")
            return None

        # Create joint states
        joints = []
        for i in range(12):
            joint = state.finger_state[i]
            joint_state = JointState(
                id=joint.id,
                position=joint.position,
                raw_position=joint.raw_position,
                sensor_id=joint.sensor_id,
                temperature=joint.temperature,
                torque=joint.torque,
                commboard_err=joint.commboard_err,
                jonitboard_err=joint.jonitboard_err,
                tipboard_err=joint.tipboard_err,
                default5=joint.default5,
                default6=joint.default6,
                default7=joint.default7,
            )
            joints.append(joint_state)

        # Create fingertip states
        fingertips = []
        for j in range(5):
            sensor_data = state.sensor_data[j]

            # Convert raw force data to Force objects
            raw_forces = [
                [force.fx, force.fy, force.fz] for force in sensor_data.raw_force
            ]

            fingertip = FingertipState(
                calc_pressure=[
                    sensor_data.calc_force.fx,
                    sensor_data.calc_force.fy,
                    sensor_data.calc_force.fz,
                ],
                raw_pressure=raw_forces,
                sensor_temperature=sensor_data.calc_temperature,
            )
            fingertips.append(fingertip)

        # Return complete hand state
        return HandState(
            joints=joints,
            fingertips=fingertips,
            error_code=error_struct.error_code,
            error_message=self.parse_error_code(error_struct)
            if error_struct.error_code != 0
            else "",
        )

    def start_reader(self):
        """
        Start the background thread that reads position at the specified frequency.
        """
        if self._reader_thread is not None and self._reader_thread.is_alive():
            print("Reader thread is already running")
            return

        self.running = True
        self._reader_thread = threading.Thread(target=self._read_loop)
        self._reader_thread.daemon = True  # Thread will exit when main program exits
        self._reader_thread.start()
        print("Position reader thread started")

    def stop_reader(self):
        """
        Stop the background thread that reads position.
        """
        self.running = False
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)  # Wait for thread to finish
            if self._reader_thread.is_alive():
                print("Warning: Reader thread did not terminate properly")
            else:
                print("Reader thread stopped")
        self._reader_thread = None

    def _read_loop(self):
        """Main reading loop that runs in a separate thread."""
        period = 1.0 / self._update_frequency  # Time period between readings

        while self.running:
            start_time = time.time()

            # Get current state
            state = self._get_current_state()

            # If we got a valid position, add it to the queue
            if state is not None:
                with self._lock:
                    self._state_queue.append(state)

            # Calculate sleep time to maintain desired frequency
            elapsed = time.time() - start_time
            sleep_time = max(0, period - elapsed)

            if sleep_time > 0:
                time.sleep(sleep_time)

    def send_command(self, command):
        print(
            f"xhand send_command result: {self._device.send_command(self._hand_id, command).error_code == 0}\n"
        )

    def write_hand_angle(self, angles, **kwargs):
        """
        Set the angles for all 12 joints of the hand.

        Parameters:
        angles (list): List of 12 angle values (in radians, depending on the system)
        **kwargs: Optional parameters like kp, ki, kd, tor_max, mode

        Returns:
        bool: True if command was sent successfully, False otherwise
        """
        if len(angles) != 12:
            print(f"Error: Expected 12 angles, got {len(angles)}")
            return False

        # Default parameters
        kp = kwargs.get("kp", 150)
        ki = kwargs.get("ki", 0)
        kd = kwargs.get("kd", 0)
        # kd = kwargs.get("kd", 500)
        tor_max = kwargs.get("tor_max", 400)
        mode = kwargs.get("mode", 3)  # Position control mode

        # Create hand command
        hand_command = xhand_control.HandCommand_t()

        # Set parameters for all 12 joints
        for i in range(12):
            hand_command.finger_command[i].id = i
            hand_command.finger_command[i].kp = kp
            hand_command.finger_command[i].ki = ki
            hand_command.finger_command[i].kd = kd
            hand_command.finger_command[i].position = angles[i]
            hand_command.finger_command[i].tor_max = tor_max
            hand_command.finger_command[i].mode = mode
            if i == 11:
                hand_command.finger_command[i].kd = 0
                hand_command.finger_command[i].kp = 100

        return hand_command


class ExoXhandSDK(XhandSDK, ExoDexterousHand):
    def __init__(
        self,
        hand_id=0,
        port="/dev/ttyUSB0",
        protocol="RS485",
        calibration_dir=None,
        per_finger_adj_val=np.zeros(12),
    ):
        super().__init__(hand_id, port, protocol)
        self.calibrate_dir = calibration_dir
        self.calibrate_angle = np.array(
            [
                # THUMB
                260 + 0,
                85,
                81,
                # INDEX
                176 + 10,
                271.5 + 5,
                84,
                # MIDDLE
                276 + 5,
                82,
                # RING
                277 + 5,
                87,
                # PINKY
                273 + 5,
                277,
            ]
        )
        if self.calibrate_dir is not None:
            self.thumb_swing_model = self.load_model(
                os.path.join(self.calibrate_dir, "joint_to_motor_index_0.pkl")
            )
            self.thumb_bend1_model = self.load_model(
                os.path.join(self.calibrate_dir, "joint_to_motor_index_1.pkl")
            )
            self.finger_bend_models = [
                self.load_model(
                    os.path.join(self.calibrate_dir, "joint_to_motor_index_4.pkl")
                ),
                self.load_model(
                    os.path.join(self.calibrate_dir, "joint_to_motor_index_6.pkl")
                ),
                self.load_model(
                    os.path.join(self.calibrate_dir, "joint_to_motor_index_8.pkl")
                ),
                self.load_model(
                    os.path.join(self.calibrate_dir, "joint_to_motor_index_10.pkl")
                ),
            ]

            self.per_finger_adj_val = per_finger_adj_val

    def predict_motor_value(self, joint_angles):
        motor_values = (joint_angles - self.calibrate_angle) / 180 * np.pi
        motor_values[0] = -motor_values[0]
        motor_values[3] = -motor_values[3]
        motor_values[4] = -motor_values[4]
        motor_values[6] = -motor_values[6]
        motor_values[8] = -motor_values[8]
        motor_values[10] = -motor_values[10]
        motor_values[11] = -motor_values[11]

        if self.calibrate_dir is not None:
            # motor_values = np.zeros(12)
            print("Oberriding...")
            motor_values[0] = self.thumb_swing_model.predict(
                [[joint_angles[0] + self.per_finger_adj_val[0]]]
            )
            motor_values[1] = self.thumb_bend1_model.predict(
                [[joint_angles[1] + self.per_finger_adj_val[1]]]
            )
            for i, model in zip([4, 6, 8, 10], self.finger_bend_models):
                motor_values[i] = model.predict(
                    [[joint_angles[i] + self.per_finger_adj_val[i]]]
                )

        # Apply clipping to the entire array
        original_value = motor_values[3]
        motor_values = np.clip(motor_values, 0, np.pi)
        motor_values[3] = original_value
        print("motor_values: ", [f"{x:.3f}" for x in motor_values])
        return motor_values


if __name__ == "__main__":
    hand = XhandSDK(port="/dev/ttyUSB0")
    hand.enumerate_devices("RS485")
    hand.connect()
    hand.start_reader()
    # hand.exam_list_hands_id()
    # for k in range(0, 2000, 5):
    #     print(f"-------------------{k}-------------------")
    #     command = hand.write_hand_angle([k / 1000] * 12)
    #     hand.send_command(command)
    #     print(hand.get_current_position().joints[0])
    #     time.sleep(0.1)
    #     # print(hand.get_current_position())

    while True:
        # test_command = [0.3] * 12
        # test_command[0] = 0.2
        # test_command[1] = 0.5
        # test_command[3] = 0
        # test_command = np.zeros(12)
        # test_command[4:] = 5 / 180 * np.pi

        # command = hand.write_hand_angle(test_command)
        # hand.send_command(command)
        time.sleep(0.1)
        # command = hand.write_hand_angle([1.5] * 12)
        # hand.send_command(command)
        # time.sleep(2)
        # print(hand.get_current_position())
        print(hand.get_tactile(calc=True))
