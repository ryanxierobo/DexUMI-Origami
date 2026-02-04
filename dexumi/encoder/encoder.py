import struct
import time
from queue import Empty, Full, Queue  # Explicitly import exceptions

from dexumi.encoder.numeric import JointFrame, Numeric
from dexumi.encoder.UARTReader import UARTReader


class JointEncoder(UARTReader):
    def __init__(
        self,
        block_size=40,
        header=bytes([0xAA, 0x55, 0x00, 0x00]),
        uart_port="/dev/ttyACM0",
        baud_rate=921600,
        verbose=True,
    ):
        super().__init__(block_size, header, uart_port, baud_rate, verbose)

    def process_buffer(self):
        while (
            len(self.buffer) >= self.block_size
        ):  # Check if the buffer has enough data to process
            # Find the start index of the first header
            start_index = self.buffer.find(self.header)

            # No header found, clear the buffer if it's longer than the block size
            if start_index == -1:
                if self.verbose:
                    print("No header found. Check the alignment of the data.")

                # If no header is found, clear the buffer if it's longer than the block size
                if len(self.buffer) > self.block_size:
                    if self.verbose:
                        print(f"Discarded data: {self.buffer[: self.block_size].hex()}")
                    self.buffer = self.buffer[-self.block_size :]
                break

            # Header found at the beginning of the buffer
            elif start_index == 0:
                # Now find the next header
                next_header_index = self.buffer.find(self.header, len(self.header))
                if next_header_index == -1:
                    # If no header is found, break and wait for more data
                    if self.verbose:
                        print(
                            "No complete block found. Waiting for more data..."
                        )  #: {self.buffer.hex()}")
                    break
                elif (next_header_index - start_index) != self.block_size:
                    # If the block is corrupted, discard the data
                    if self.verbose:
                        print(
                            f"Corrupted block found: {self.buffer[:next_header_index].hex()}"
                        )
                    self.buffer = self.buffer[next_header_index:]
                else:
                    # We found a complete block
                    block = self.buffer[start_index:next_header_index]
                    self.process_block(block)

                    # Remove the processed block from the buffer
                    self.buffer = self.buffer[next_header_index:]

            # Header found in the middle of the buffer, discard the data before the header
            else:
                if self.verbose:
                    print(f"Discarded data: {self.buffer[:start_index].hex()}")
                self.buffer = self.buffer[start_index:]

    def process_block(self, block):
        # Unpack the block (skip the header)
        data = struct.unpack("<8iI", block[4:])  # 8 int32_t + 1 uint32_t CRC

        # Extract the 16 int32_t values and CRC
        data_values = data[:-1]
        crc = data[-1]

        # Convert data_values to voltages
        voltages = [(float(val) * 5.0 / 0x7FFFFF) for val in data_values]

        # Split CRC into checksum and timestamp
        checksum = (crc >> 16) & 0xFFFF
        timestamp = crc & 0xFFFF

        receive_time = time.monotonic()

        # Calculate the expected checksum (16-bit sum of data_values)
        expected_checksum = sum(data_values) & 0xFFFF

        # Calculate the joint angles
        reference_voltage = voltages[-2]
        joint_angles = [
            val / reference_voltage * 360 for val in voltages if reference_voltage != 0
        ]
        joint_angles = joint_angles[:-2]  # Skip the last two values

        frame_data = JointFrame(
            capture_time=receive_time,
            receive_time=receive_time,
            joint_angles=joint_angles,
            raw_voltage=voltages,
        )
        self.put_frame(frame_data)

        if self.verbose:
            print(f"Voltages: {[round(val, 2) for val in voltages]}")
            print(f"Joint Angle: {[round(val, 2) for val in joint_angles]}")
            print(f"val: {round(joint_angles[0], 2)}")
            print(f"Timestamp: {int(timestamp)}")
            print(f"Checksum: {(checksum)} (Expected: {(expected_checksum)})")
            print(f"Checksum Valid: {checksum == expected_checksum}")
            print("-" * 40)


class InspireEncoder(JointEncoder, Numeric):
    def __init__(
        self,
        device_name: str = "inspire_hand",
        latency: float = 0.000,
        block_size=40,
        header=bytes([0xAA, 0x55, 0x00, 0x00]),
        uart_port="/dev/ttyACM0",
        baud_rate=921600,
        verbose=True,
    ):
        JointEncoder.__init__(self, block_size, header, uart_port, baud_rate, verbose)
        Numeric.__init__(self, device_name, latency)

    def start_streaming(self):
        self.start()

    def stop_streaming(self):
        self.stop()

    def get_numeric_frame(self):
        with self.frame_queue.mutex:
            if self.frame_queue.queue:
                return self.frame_queue.queue[-1]


class XhandEncoder(JointEncoder, Numeric):
    def __init__(
        self,
        device_name: str = "xhand",
        latency: float = 0.000,
        block_size=64,
        header=bytes([0xAA, 0x55, 0x00, 0x00]),
        uart_port="/dev/ttyACM0",
        baud_rate=921600,
        verbose=True,
    ):
        JointEncoder.__init__(self, block_size, header, uart_port, baud_rate, verbose)
        Numeric.__init__(self, device_name, latency)

    def start_streaming(self):
        self.start()

    def stop_streaming(self):
        self.stop()

    def get_numeric_frame(self):
        with self.frame_queue.mutex:
            if self.frame_queue.queue:
                return self.frame_queue.queue[-1]

    def process_block(self, block):
        # Unpack the block (skip the header)
        data = struct.unpack("<14iI", block[4:])  # 14 int32_t + 1 uint32_t CRC

        # Extract the 14 int32_t values and CRC
        data_values = data[:-1]
        crc = data[-1]

        # Convert data_values to voltages
        voltages = [(float(val) * 5.0 / 0x7FFFFF) for val in data_values]

        # Split CRC into checksum and timestamp
        checksum = (crc >> 16) & 0xFFFF
        timestamp = crc & 0xFFFF

        receive_time = time.monotonic()

        # Calculate the expected checksum (16-bit sum of data_values)
        expected_checksum = sum(data_values) & 0xFFFF

        # Calculate the joint angles
        reference_voltage = voltages[-2]
        joint_angles = [
            val / reference_voltage * 360 for val in voltages if reference_voltage != 0
        ]
        joint_angles = joint_angles[:-2]  # Skip the last two values

        frame_data = JointFrame(
            capture_time=receive_time,
            receive_time=receive_time,
            joint_angles=joint_angles,
            raw_voltage=voltages,
        )
        self.put_frame(frame_data)

        if self.verbose:
            print(f"Voltages: {[round(val, 3) for val in voltages]}")
            print(f"Joint Angle: {[round(val, 3) for val in joint_angles]}")
            print(f"val: {round(joint_angles[1], 5)}")
            print(f"Timestamp: {int(timestamp)}")
            print(f"Checksum: {(checksum)} (Expected: {(expected_checksum)})")
            print(f"Checksum Valid: {checksum == expected_checksum}")
            print("-" * 40)


if __name__ == "__main__":
    encoder = XhandEncoder(uart_port="/dev/ttyACM0")
    encoder.start_streaming()
    # time.sleep(5)
    # encoder.stop_streaming()
    # print(encoder.get_numeric_frame())
