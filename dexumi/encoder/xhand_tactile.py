import struct
import time

import numpy as np
from dexumi.encoder.numeric import FSRFrame, Numeric
from dexumi.encoder.UARTReader import UARTReader


class XhandUARTReader(UARTReader):
    def __init__(
        self,
        uart_port="/dev/ttyACM0",
        header=bytes([0x55, 0xAA]),
        block_size=728,  # Total size including header, data, and CRC
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
                print("No header found. Check the alignment of the data.")

                # If no header is found, clear the buffer if it's longer than the block size
                if len(self.buffer) > self.block_size:
                    print(f"Discarded data: {self.buffer[: self.block_size].hex()}")
                    self.buffer = self.buffer[-self.block_size :]
                break

            # Header found at the beginning of the buffer
            elif start_index == 0:
                # Now find the next header
                next_header_index = self.buffer.find(self.header, len(self.header))
                if next_header_index == -1:
                    # If no header is found, break and wait for more data
                    # print(f"No complete block found. Waiting for more data...")#: {self.buffer.hex()}")
                    break
                elif (next_header_index - start_index) != self.block_size:
                    # If the block is corrupted, discard the data
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
                print(f"Discarded data: {self.buffer[:start_index].hex()}")
                self.buffer = self.buffer[start_index:]

    def process_block(self, block):
        # Unpack the block (skip the header)
        # 1 int8_t fx, 1 int8_t fy, 1 uint8_t fz, 120*3 uint16_t force, 1 uint16_t timestamp, and 1 uint8_t checksum
        # unpack_format = (
        #     "<" + "2b B" + "120H " * 3 + "H B"
        # )  # Expands to 2b, B, 3 sets of 120H, followed by H, B
        unpack_format = (
            "<" + "2b B" + "H B"
        )  # Expands to 2b, B, 3 sets of 120H, followed by H, B
        data = struct.unpack(unpack_format, block[2:])

        # print(data)
        # Extract force
        force_xyz = np.array(data[0:3])
        total_force = np.linalg.norm(
            force_xyz
        )  # Equivalent to sqrt(fx^2 + fy^2 + fz^2)
        # force_matrix_x = np.array(data[3:123], dtype=float).reshape(10, 12) / 100.0
        # force_matrix_y = np.array(data[123:243], dtype=float).reshape(10, 12) / 100.0
        # force_matrix_z = np.array(data[243:363], dtype=float).reshape(10, 12) / 100.0

        # Extract the timestamp and checksum
        timestamp = data[-2]
        checksum = data[-1]

        # Calculate the expected checksum (8-bit sum of data)
        expected_checksum = sum(block[2:-1]) & 0xFF

        # Store in dictionary
        force_data = {
            "finger": {
                "force_xyz": force_xyz,
                "total_force": total_force,
                # "force_matrix_x": force_matrix_x,
                # "force_matrix_y": force_matrix_y,
                # "force_matrix_z": force_matrix_z,
            },
            "timestamp": timestamp,
            "checksum_valid": (checksum == expected_checksum),
        }
        frame_data = FSRFrame(
            capture_time=time.monotonic(),
            receive_time=time.monotonic(),
            fsr_values=force_data["finger"]["force_xyz"],
        )
        # print(force_data["finger"]["force_xyz"])
        self.put_frame(frame_data)
        if self.verbose:
            print(f"Timestamp: {int(timestamp)}")
            print(f"Checksum: {(checksum)} (Expected: {(expected_checksum)})")
            print(f"Checksum Valid: {checksum == expected_checksum}")
            print("-" * 40)


class XhandTactile(XhandUARTReader, Numeric):
    def __init__(
        self,
        device_name: str = "xhand_fsr",
        latency: float = 0.000,
        uart_port="/dev/ttyACM0",
        header=bytes([85, 170]),
        block_size=8,
        baud_rate=921600,
        verbose=True,
    ):
        XhandUARTReader.__init__(
            self, uart_port, header, block_size, baud_rate, verbose
        )
        Numeric.__init__(self, device_name, latency)

    def start_streaming(self):
        self.start()

    def stop_streaming(self):
        self.stop()


if __name__ == "__main__":
    encoder = XhandTactile(
        uart_port="/dev/ttyACM3",
    )
    encoder.start_streaming()
