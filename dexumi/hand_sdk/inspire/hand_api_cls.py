import pickle
import struct
import threading
import time

import numpy as np
import serial

from dexumi.hand_sdk.dexhand import DexterousHand, ExoDexterousHand


class InspireSDK(DexterousHand):
    # Command constants
    kSend_Frame_Head1 = 0xEB
    kSend_Frame_Head2 = 0x90
    kRcv_Frame_Head1 = 0x90
    kRcv_Frame_Head2 = 0xEB

    # Command types
    kCmd_Handg3_Read = 0x11  # read register
    kCmd_Handg3_Write = 0x12  # write register
    kCmd_Mc_Angle_Force = 0x14  # motor motion enable
    kCmd_Mc_Force = 0x15  # motor motion disable
    kCmd_Mc_Current = 0x16  # save parameters to flash
    kCmd_Mc_All = 0x17  # get drive status

    header = bytes([0x90, 0xEB])
    block_size = 20  # Total size including header, data, and CRC

    def __init__(
        self,
        hand_id=0x01,
        port="/dev/ttyUSB0",
        baudrate=115200,
        databits=8,
        parity="N",
        stopbits=1,
        read_rate=60,  # Hz
        verbose=False,  # Add verbose parameter
    ):
        """Initialize the Hand SDK with serial configuration."""
        self.hand_id = hand_id
        self.port = port
        self.baudrate = baudrate
        self.databits = databits
        self.parity = parity
        self.stopbits = stopbits
        self.ser = None
        self.read_rate = read_rate
        self.verbose = verbose

        # UART reader attributes
        self.buffer = bytearray()
        self.running = False
        self.reader_thread = None
        self.write_lock = threading.Lock()

        # Position tracking
        self.current_position = None
        self.position_lock = threading.Lock()

    def _debug(self, message):
        """Helper method for debug output."""
        if self.verbose:
            print(message)

    def connect(self):
        """Configure and connect to the serial port."""
        try:
            self.ser = serial.Serial(self.port)
            self.ser.baudrate = self.baudrate

            # Set data bits
            if self.databits == 8:
                self.ser.bytesize = serial.EIGHTBITS
            elif self.databits == 7:
                self.ser.bytesize = serial.SEVENBITS

            # Set parity
            if self.parity == "N":
                self.ser.parity = serial.PARITY_NONE
            elif self.parity == "E":
                self.ser.parity = serial.PARITY_EVEN
            elif self.parity == "O":
                self.ser.parity = serial.PARITY_ODD

            # Set stop bits
            if self.stopbits == 1:
                self.ser.stopbits = serial.STOPBITS_ONE
            elif self.stopbits == 2:
                self.ser.stopbits = serial.STOPBITS_TWO

            self.ser.timeout = 0.1
            if self.verbose:
                print(
                    f"Serial port {self.port} configured: baudrate={self.baudrate}, "
                    f"databits={self.databits}, parity={self.parity}, stopbits={self.stopbits}"
                )
            return True

        except serial.SerialException as e:
            print(f"Error configuring serial port: {e}")
            return False

    def disconnect(self):
        """Close the serial connection and stop the reader thread."""
        self.stop_reader()
        if self.ser and self.ser.is_open:
            self.ser.close()

    def get_current_position(self):
        """Thread-safe method to get the current hand position with timeout."""
        timeout = 0.1  # 100ms timeout
        start_time = time.time()

        while True:
            with self.position_lock:
                if self.current_position is not None:
                    return self.current_position[::-1]

            if time.time() - start_time > timeout:
                return None

            time.sleep(0.01)  # Small sleep to prevent CPU spinning

    def start_reader(self):
        """Start the UART reader thread."""
        if not self.running:
            self.running = True
            self.reader_thread = threading.Thread(target=self._read_loop)
            self.reader_thread.daemon = True
            self.reader_thread.start()
            print("UART reader started")

    def stop_reader(self):
        """Stop the UART reader thread."""
        if self.running:
            self.running = False
            if self.reader_thread:
                self.reader_thread.join()
            print("UART reader stopped")

    def _read_loop(self):
        """Main reading loop that runs in a separate thread."""
        while self.running:
            try:
                # Send position read command
                read_cmd = self.build_read_hand_register_data(0x060A, 0x0C)

                with self.write_lock:
                    bytes_written = self.ser.write(read_cmd)
                    self._debug(f"Wrote {bytes_written} bytes")

                # Wait a bit for the response
                time.sleep(0.75 * (1 / self.read_rate))
                # Read available data
                if self.ser.in_waiting:
                    data = self.ser.read(self.ser.in_waiting)
                    if data:
                        self.buffer.extend(data)
                        self._process_buffer()
                else:
                    self._debug("No data available to read")

                time.sleep(1 / self.read_rate)

            except Exception as e:
                print(f"Error in read loop: {e}")
                import traceback

                traceback.print_exc()
                time.sleep(0.1)

    def _process_buffer(self):
        """Process the received data buffer."""
        self._debug(f"\nProcessing buffer of length {len(self.buffer)}")
        self._debug(f"Buffer content: {[f'{b:02x}' for b in self.buffer]}")

        while len(self.buffer) >= self.block_size:
            # Find header
            start_index = self.buffer.find(self.header)
            self._debug(f"Looking for header {[f'{b:02x}' for b in self.header]}")
            self._debug(f"Header search result: {start_index}")

            if start_index == -1:
                # No header found, clear old data
                if len(self.buffer) > self.block_size:
                    self._debug("No header found, clearing old data")
                    self._debug(
                        f"Discarding: {[f'{b:02x}' for b in self.buffer[: -self.block_size]]}"
                    )
                    self.buffer = self.buffer[-self.block_size :]
                break

            elif start_index > 0:
                # Discard data before header
                self._debug(
                    f"Found header at position {start_index}, discarding preceding data"
                )
                self._debug(
                    f"Discarding: {[f'{b:02x}' for b in self.buffer[:start_index]]}"
                )
                self.buffer = self.buffer[start_index:]
                continue

            # Check if we have enough data for a complete block
            if len(self.buffer) < self.block_size:
                self._debug(
                    f"Not enough data for complete block. Have {len(self.buffer)} bytes, need {self.block_size}"
                )
                break

            # Process the block
            block = self.buffer[: self.block_size]

            if self._validate_block(block):
                self._debug("Block validation successful")
                self._handle_block(block)
            else:
                self._debug("Block validation failed")

            # Remove processed block
            self.buffer = self.buffer[self.block_size :]
            self._debug(f"Remaining buffer size: {len(self.buffer)}")

    def _validate_block(self, block):
        """Validate a data block."""
        try:
            if len(block) < self.block_size:
                self._debug("Block too short")
                return False

            if block[0] != self.kRcv_Frame_Head1 or block[1] != self.kRcv_Frame_Head2:
                self._debug("Invalid header")
                return False

            if block[2] != self.hand_id:
                self._debug(f"Invalid ID: {block[2]} != {self.hand_id}")
                return False

            if block[4] != self.kCmd_Handg3_Read:
                self._debug(f"Invalid command: {block[4]:02x}")
                return False

            expected_checksum = sum(block[2:-1]) & 0xFF
            actual_checksum = block[-1]

            if expected_checksum != actual_checksum:
                self._debug(
                    f"Checksum mismatch: expected {expected_checksum:02x}, got {actual_checksum:02x}"
                )
                return False

            return True

        except Exception as e:
            self._debug(f"Error validating block: {e}")
            return False

    def _handle_block(self, block):
        """Handle a validated data block."""
        try:
            # Check if we have a response to our position read command
            reg_addr = block[5] | (block[6] << 8)
            if reg_addr != 0x060A:
                self._debug(f"Unexpected register address: {reg_addr:04x}")
                return

            # Extract position data (6 16-bit values)
            position_data = struct.unpack("<6H", block[7:-1])

            with self.position_lock:
                self.current_position = position_data

            # if self.current_position:
            #     self._debug(f"Updated position: {self.current_position}")

        except Exception as e:
            self._debug(f"Error handling block: {e}")
            import traceback

            traceback.print_exc()

    def send_command(self, command):
        """Thread-safe method to send a command to the hand."""
        with self.write_lock:
            try:
                self.ser.write(command)
            except Exception as e:
                self._debug(f"Error sending command: {e}")

    def write_hand_angle(self, val1, val2, val3, val4, val5, val6):
        """Write angles to the hand actuator."""
        command = bytearray(20)
        m_unChecksum = 0

        command[0] = self.kSend_Frame_Head1
        command[1] = self.kSend_Frame_Head2
        command[2] = self.hand_id
        command[3] = 0x0F
        command[4] = self.kCmd_Handg3_Write
        command[5] = 0xCE
        command[6] = 0x05
        command[7] = val1 & 0xFF
        command[8] = val1 >> 8
        command[9] = val2 & 0xFF
        command[10] = val2 >> 8
        command[11] = val3 & 0xFF
        command[12] = val3 >> 8
        command[13] = val4 & 0xFF
        command[14] = val4 >> 8
        command[15] = val5 & 0xFF
        command[16] = val5 >> 8
        command[17] = val6 & 0xFF
        command[18] = val6 >> 8

        for i in range(2, 19):
            m_unChecksum += command[i]
        command[19] = m_unChecksum & 0xFF

        # self.send_command(command)
        return command

    def write_hand_angle_force(self, val1, val2, val3, val4, val5, val6):
        """Write angles with force to the hand actuator."""
        command = bytearray(28)
        m_unChecksum = 0
        _force = 1000

        command[0] = self.kSend_Frame_Head1
        command[1] = self.kSend_Frame_Head2
        command[2] = self.hand_id
        command[3] = 0x17
        command[4] = 0x20
        command[5] = val1 & 0xFF
        command[6] = val1 >> 8
        command[7] = val2 & 0xFF
        command[8] = val2 >> 8
        command[9] = val3 & 0xFF
        command[10] = val3 >> 8
        command[11] = val4 & 0xFF
        command[12] = val4 >> 8
        command[13] = val5 & 0xFF
        command[14] = val5 >> 8
        command[15] = val6 & 0xFF
        command[16] = val6 >> 8

        # Set force values
        for i in range(17, 27, 2):
            command[i] = _force & 0xFF
            command[i + 1] = _force >> 8

        for i in range(2, 27):
            m_unChecksum += command[i]
        command[27] = m_unChecksum & 0xFF

        # self.send_command(command)
        return command

    def build_read_hand_register_data(self, reg_addr, read_len):
        """Build command to read hand register data."""
        command = bytearray(9)
        m_unChecksum = 0

        command[0] = self.kSend_Frame_Head1
        command[1] = self.kSend_Frame_Head2
        command[2] = self.hand_id
        command[3] = 0x04
        command[4] = self.kCmd_Handg3_Read
        command[5] = reg_addr & 0xFF
        command[6] = reg_addr >> 8
        command[7] = read_len

        for i in range(2, 8):
            m_unChecksum += command[i]
        command[8] = m_unChecksum & 0xFF

        return command


class ExoInspireSDK(InspireSDK, ExoDexterousHand):
    def __init__(
        self,
        finger_mapping_model_path,
        thumb_swing_model_path=None,
        thumb_middle_model_path=None,
        per_finger_adj_val=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.thumb_swing_model_path = thumb_swing_model_path
        self.thumb_middle_model_path = thumb_middle_model_path
        self.finger_mapping_model_path = finger_mapping_model_path

        self.thumb_swing_model = self.load_model(self.thumb_swing_model_path)
        self.thumb_middle_model = self.load_model(self.thumb_middle_model_path)
        self.finger_mapping_model = self.load_model(self.finger_mapping_model_path)

        self.per_finger_adj_val = per_finger_adj_val

    def predict_motor_value(self, joint_angle: np.ndarray) -> np.ndarray:
        """Predict motor value from encoder joint angle."""
        predicted_motor_value = np.zeros(6)
        if self.thumb_swing_model:
            thumb_swing_angle = joint_angle[0]
            d = -(thumb_swing_angle - 173.5 + self.per_finger_adj_val[0]) / 180 * np.pi
            d = np.clip(d, 0, 2)
            predicted_finger_motor = self.thumb_swing_model.predict([[d]])
            predicted_motor_value[0] = predicted_finger_motor * 1000
        else:
            raise ValueError("Thumb swing model not loaded.")

        if self.thumb_middle_model:
            thumb_middle_angle = joint_angle[1]
            # train
            d = -(thumb_middle_angle - 237.5 + self.per_finger_adj_val[1]) / 180 * np.pi

            # print("d:", d)
            d = np.clip(d, 0, 2)
            predicted_finger_motor = self.thumb_middle_model.predict([[d]])
            predicted_motor_value[1] = predicted_finger_motor * 1000
        else:
            raise ValueError("Thumb middle model not loaded.")

        if self.finger_mapping_model:
            for i, agl in enumerate(joint_angle):
                if i >= 2:
                    if self.per_finger_adj_val is not None:
                        d = agl + self.per_finger_adj_val[i]
                        predicted_finger_motor = self.finger_mapping_model.predict(
                            [[d]]
                        )
                    else:
                        raise ValueError("Reference value not provided.")
                    predicted_motor_value[i] = predicted_finger_motor
        else:
            raise ValueError("Finger mapping model not loaded.")
        predicted_motor_value = predicted_motor_value.astype(np.int32)
        predicted_motor_value = np.clip(predicted_motor_value, 0, 1000)
        return predicted_motor_value


# Example usage
if __name__ == "__main__":
    hand = InspireSDK()
    if hand.connect():
        hand.run_control_loop()
