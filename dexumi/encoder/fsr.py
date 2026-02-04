import struct
import time

from dexumi.encoder.numeric import FSRFrame, Numeric
from dexumi.encoder.UARTReader import UARTReader


class FSRUARTReader(UARTReader):
    def __init__(
        self,
        uart_port="/dev/ttyACM1",
        header=bytes([0xAA, 0x55, 0x00, 0x00]),
        block_size=60,  # Total size including header, data, and CRC
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
                    print(f"Discarded data: {self.buffer[: self.block_size].hex()}")
                    self.buffer = self.buffer[-self.block_size :]
                break

            # Header found at the beginning of the buffer
            elif start_index == 0:
                # Now find the next header
                next_header_index = self.buffer.find(self.header, len(self.header))
                if next_header_index == -1:
                    if self.verbose:
                        print(
                            "No complete block found. Waiting for more data..."
                        )  #: {self.buffer.hex()}")
                    break
                elif (next_header_index - start_index) != self.block_size:
                    # If the block is corrupted, discard the data
                    if self.verbose:
                        print(
                            f"Corrupted block found in fsr: {self.buffer[:next_header_index].hex()}"
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
        data = struct.unpack("<12iII", block[4:])  # 12 int32_t values, 2 uint32_t value

        # Extract the 16 int32_t values and CRC
        data_values = data[:-1]
        crc = data[-1]

        # Convert data_values to voltages
        voltages = [(float(val) * 5.0 / 0x7FFFFF) for val in data_values[:-2]]

        # Extract the FSR values
        packed_value = data_values[-1]
        fsr = [packed_value & 0xFFFF, (packed_value >> 16) & 0xFFFF]

        # Split CRC into checksum and timestamp
        checksum = (crc >> 16) & 0xFFFF
        timestamp = crc & 0xFFFF

        # Calculate the expected checksum (16-bit sum of data_values)
        expected_checksum = sum(data_values) & 0xFFFF

        frame_data = FSRFrame(
            capture_time=time.monotonic(),
            receive_time=time.monotonic(),
            fsr_values=fsr,
        )
        self.put_frame(frame_data)

        if self.verbose:
            print(f"Voltages: {[round(val, 2) for val in voltages]}")
            print(f"FSR: {[round(val, 2) for val in fsr]}")
            print(f"Timestamp: {int(timestamp)}")
            print(f"Checksum: {(checksum)} (Expected: {(expected_checksum)})")
            print(f"Checksum Valid: {checksum == expected_checksum}")
            print("-" * 40)


class FSRSensor(FSRUARTReader, Numeric):
    def __init__(
        self,
        device_name: str = "inspire_fsr",
        latency: float = 0.000,
        uart_port="/dev/ttyACM1",
        header=bytes([0xAA, 0x55, 0x00, 0x00]),
        block_size=60,  # Total size including header, data, and CRC
        baud_rate=921600,
        verbose=True,
    ):
        FSRUARTReader.__init__(self, uart_port, header, block_size, baud_rate, verbose)
        Numeric.__init__(self, device_name, latency)

    def start_streaming(self):
        self.start()

    def stop_streaming(self):
        self.stop()


if __name__ == "__main__":
    encoder = FSRSensor()
    encoder.start_streaming()
    # time.sleep(5)
    # encoder.stop_streaming()
    # print(encoder.get_numeric_frame())
