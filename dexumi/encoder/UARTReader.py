import threading
import time
from abc import abstractmethod
from queue import Empty, Full, Queue  # Explicitly import exceptions
import serial


class UARTReader:
    def __init__(
        self,
        block_size,
        header,
        uart_port="/dev/ttyACM0",
        baud_rate=921600,
        verbose=True,
    ):
        self.block_size = block_size
        self.header = header
        self.buffer = bytearray()
        self.running = True
        self.serial_port = serial.Serial(uart_port, baud_rate, timeout=1)
        self.verbose = verbose
        self.frame_queue = Queue(maxsize=20)

    def read_from_uart(self):
        while self.running:
            if self.serial_port.in_waiting:  # Check if there's data first
                data = self.serial_port.read(self.serial_port.in_waiting)
                self.buffer.extend(data)
                self.process_buffer()
            else:
                time.sleep(0.001)  # Small sleep to prevent CPU thrashing

    @abstractmethod
    def process_buffer(self):
        pass

    @abstractmethod
    def process_block(self):
        pass

    def start(self):
        print("UART reader started")
        self.thread = threading.Thread(target=self.read_from_uart)
        self.thread.start()

    def stop(self):
        print("Closing UART port...")
        self.running = False
        self.thread.join()
        self.serial_port.close()

    def put_frame(self, frame_data):
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()  # Remove oldest frame
                if self.verbose:
                    print("Queue full, removed oldest frame")
            except Empty:
                pass

        try:
            self.frame_queue.put_nowait(frame_data)
            if self.verbose:
                print("Put frame queued")
                print("queue size", self.frame_queue.qsize())
        except Full:
            if self.verbose:
                print("Failed to put frame")

    def get_numeric_frame(self):
        with self.frame_queue.mutex:
            if self.frame_queue.queue:
                return self.frame_queue.queue[-1]
