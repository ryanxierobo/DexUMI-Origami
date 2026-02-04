import pickle
import time
import traceback
import uuid
from dataclasses import dataclass
from enum import Enum
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Any, Optional
from urllib.parse import urlparse

import zmq

from ..ring_buffer import RingBuffer


class TransportType(Enum):
    IPC = "ipc"
    TCP = "tcp"


class RequestType(Enum):
    """Base enum for request types. Inherit and extend this in your implementation."""

    pass


@dataclass
class Request:
    """Generic request container."""

    type: RequestType
    params: dict = None
    request_id: str = None

    def __post_init__(self):
        if self.request_id is None:
            self.request_id = str(uuid.uuid4())


@dataclass
class Response:
    """Generic response container."""

    data: Any
    request_id: str


class ZMQServerBase:
    def __init__(
        self,
        pub_address: str,
        req_address: str,
        topic: str = "data",
        max_buffer_size: int = 30,
        pub_frequency: int = 60,
        req_frequency: int = 1000,
        frames_per_publish: int = 1,
        verbose=True,
    ):
        """
        Initialize ZMQ server with flexible transport support.

        Args:
            pub_address: Full address for PUB socket (e.g., "ipc:///tmp/pub.ipc" or "tcp://127.0.0.1:5555")
            req_address: Full address for REQ socket (e.g., "ipc:///tmp/req.ipc" or "tcp://127.0.0.1:5556")
        """
        self.verbose = verbose
        # Convert topic to bytes
        self.topic = topic.encode() if isinstance(topic, str) else topic

        # Parse addresses
        self.pub_transport = TransportType(urlparse(pub_address).scheme)
        self.req_transport = TransportType(urlparse(req_address).scheme)

        # PUB socket setup
        self.pub_context = zmq.Context()
        self.pub_socket = self.pub_context.socket(zmq.PUB)
        self.pub_socket.setsockopt(zmq.SNDHWM, 1)
        self.pub_socket.setsockopt(zmq.LINGER, 0)
        self.pub_socket.bind(pub_address)

        # ROUTER socket setup
        self.router_context = zmq.Context()
        self.router_socket = self.router_context.socket(zmq.ROUTER)
        self.router_socket.setsockopt(zmq.LINGER, 0)
        self.router_socket.bind(req_address)

        self.data_buffer = RingBuffer(max_buffer_size)
        self.frames_per_publish = max(1, min(frames_per_publish, max_buffer_size))
        self.running = False
        self.publish_thread = None
        self.request_thread = None

        self.pub_frequency = pub_frequency
        self.req_frequency = req_frequency

        # Add pause-related attributes
        self._paused = False
        self._pause_lock = Lock()

    def is_paused(self) -> bool:
        """Check if the server is currently paused."""
        with self._pause_lock:
            return self._paused

    def pause(self):
        """Pause data publication."""
        with self._pause_lock:
            self._paused = True

    def resume(self):
        """Resume data publication."""
        with self._pause_lock:
            self._paused = False

    def _publish_loop(self):
        """Main publishing loop with K-frame support and pause capability."""
        while self.running:
            with self._rate_limit(self.pub_frequency):
                # Check pause state
                if not self.is_paused():
                    data = self._get_data()
                    if data is not None:
                        # Write to ring buffer
                        self.data_buffer.write(data)

                        # Get last K frames
                        frames = self.data_buffer.read_last(self.frames_per_publish)
                        # Package frames with metadata
                        publish_data = frames
                        self.pub_socket.send_multipart(
                            [self.topic, pickle.dumps(publish_data)]
                        )

    def start(self):
        """Start the publish and request handling threads."""
        self.running = True
        self._paused = False  # Ensure we start in non-paused state

        self.publish_thread = Thread(target=self._publish_loop)
        self.publish_thread.daemon = False
        self.publish_thread.start()

        self.request_thread = Thread(target=self._handle_requests)
        self.request_thread.daemon = False
        self.request_thread.start()

    def _debug(self, message: str):
        """Print debug message if verbose mode is enabled."""
        if self.verbose:
            print(f"[SERVER DEBUG] {message}")

    def _handle_requests(self):
        """Handle incoming requests with message part debugging and rate limiting."""
        last_request_time = time.time()
        while self.running:
            # Apply rate limiting using the existing context manager
            with self._rate_limit(self.req_frequency):
                try:
                    if self.router_socket.poll(timeout=0, flags=zmq.POLLIN):
                        # Get all message parts first
                        current_time = time.time()
                        message_parts = self.router_socket.recv_multipart()
                        self._debug(f"Received {len(message_parts)} message parts")
                        self._debug(
                            f"Time since last request: {current_time - last_request_time}"
                        )
                        self._debug(
                            f"Socket events: {self.router_socket.getsockopt(zmq.EVENTS)}"
                        )
                        last_request_time = current_time

                        # Debug print each part
                        for i, part in enumerate(message_parts):
                            try:
                                if i == 0:
                                    self._debug(f"Part {i} (identity): {part.hex()}")
                                else:
                                    # Try to decode if it's text, otherwise show as hex
                                    try:
                                        decoded = part.decode("utf-8")
                                        self._debug(f"Part {i}: '{decoded}'")
                                    except UnicodeDecodeError:
                                        self._debug(f"Part {i}: {part.hex()}")
                            except Exception as e:
                                self._debug(f"Error debugging part {i}: {str(e)}")

                        # Ensure we have exactly 3 parts
                        if len(message_parts) != 3:
                            self._debug(
                                "Invalid message format - wrong number of parts"
                            )
                            # Try to get the identity from the first part if available
                            if message_parts:
                                identity = message_parts[0]
                                error_response = Response(
                                    data={"error": "Invalid message format"},
                                    request_id=str(uuid.uuid4()),
                                )
                                self.router_socket.send_multipart(
                                    [identity, b"", pickle.dumps(error_response)]
                                )
                            continue

                        # Now safely unpack the parts
                        identity, empty, request_msg = message_parts

                        try:
                            request = pickle.loads(request_msg)
                            self._debug(
                                f"Unpickled request with ID: {getattr(request, 'request_id', 'unknown')}"
                            )

                            if not isinstance(request, Request):
                                response = Response(
                                    data={"error": "Invalid request format"},
                                    request_id=getattr(
                                        request, "request_id", "unknown"
                                    ),
                                )
                            else:
                                result = self._process_request(request)
                                response = Response(
                                    data=result, request_id=request.request_id
                                )

                            self._debug(
                                f"Sending response for request ID: {response.request_id}"
                            )
                            self.router_socket.send_multipart(
                                [identity, b"", pickle.dumps(response)]
                            )
                            self._debug(
                                f"Response sent successfully for request ID: {response.request_id}"
                            )

                        except Exception as e:
                            self._debug(f"Error processing request: {str(e)}")

                except Exception as e:
                    self._debug(f"Critical error in request handler: {str(e)}")
                    self._debug(f"Traceback: {traceback.format_exc()}")

    def get_last_n_data(self, n: int) -> list:
        """Get the last n items from the buffer."""
        return self.data_buffer.read_last(n)

    def clear_buffer(self):
        """Clear the data buffer."""
        self.data_buffer.clear()

    def _get_data(self) -> Any:
        """Override this method to provide data to publish."""
        raise NotImplementedError("Subclass must implement _get_data()")

    def _process_request(self, request: Request) -> Any:
        """Override this method to process incoming requests."""
        raise NotImplementedError("Subclass must implement _process_request()")

    @staticmethod
    def _rate_limit(frequency):
        """Context manager for rate limiting."""

        class RateContext:
            def __init__(self, freq):
                self.period = 1.0 / freq if freq > 0 else 0
                self.start_time = None

            def __enter__(self):
                self.start_time = time.time()
                return self

            def __exit__(self, *args):
                if self.period > 0:
                    elapsed = time.time() - self.start_time
                    if elapsed < self.period:
                        time.sleep(self.period - elapsed)

        return RateContext(frequency)

    def stop(self):
        """Stop all threads and clean up."""
        self.running = False
        if self.publish_thread:
            self.publish_thread.join()
        if self.request_thread:
            self.request_thread.join()
        self.pub_socket.close()
        self.router_socket.close()
        self.pub_context.term()
        self.router_context.term()


class ZMQClientBase:
    def __init__(
        self,
        pub_address: str,
        req_address: str,
        topic: str = "data",
        req_frequency: int = 1000,
        verbose: bool = True,
    ):
        """Initialize ZMQ client with message queue for controlled sending."""
        self.verbose = verbose
        self.topic = topic.encode() if isinstance(topic, str) else topic
        self.pub_transport = TransportType(urlparse(pub_address).scheme)
        self.req_transport = TransportType(urlparse(req_address).scheme)
        self.req_address = req_address
        self.req_frequency = req_frequency

        # SUB socket setup
        self.sub_context = zmq.Context()
        self.sub_socket = self.sub_context.socket(zmq.SUB)
        self.sub_socket.setsockopt(zmq.RCVHWM, 1)
        self.sub_socket.setsockopt(zmq.LINGER, 0)
        self.sub_socket.connect(pub_address)
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, self.topic)

        # DEALER socket setup
        self.dealer_context = zmq.Context()
        self.dealer_socket = self.dealer_context.socket(zmq.DEALER)
        self.dealer_socket.setsockopt(zmq.LINGER, 0)
        self.dealer_socket.connect(req_address)

        # Initialize response handling with locks
        self.pending_responses = {}
        self.pending_responses_lock = Lock()
        self.running = True

        # Add request tracking
        self.active_requests = set()
        self.active_requests_lock = Lock()

        # Initialize send queue and thread
        self.send_queue = Queue()
        self.send_thread = Thread(target=self._send_loop)
        self.send_thread.daemon = True

        # Start response handler thread
        self.response_thread = Thread(target=self._handle_responses)
        self.response_thread.daemon = True

        # Start threads
        self.send_thread.start()
        self.response_thread.start()

        self._debug("ZMQClientBase initialized")

    def _debug(self, message: str):
        """Print debug message if verbose mode is enabled."""
        if self.verbose:
            print(f"[DEBUG] {message}")

    def _error(self, message: str):
        """Print error message regardless of verbose mode."""
        print(f"[ERROR] {message}")

    @staticmethod
    def _rate_limit(frequency):
        """Context manager for rate limiting."""

        class RateContext:
            def __init__(self, freq):
                self.period = 1.0 / freq if freq > 0 else 0
                self.start_time = None

            def __enter__(self):
                self.start_time = time.time()
                return self

            def __exit__(self, *args):
                if self.period > 0:
                    elapsed = time.time() - self.start_time
                    if elapsed < self.period:
                        time.sleep(self.period - elapsed)

        return RateContext(frequency)

    def _send_loop(self):
        """Process send queue with rate limiting."""
        while self.running:
            with self._rate_limit(self.req_frequency):
                try:
                    request = self.send_queue.get(timeout=0)
                    self._debug(f"Sending request from queue: {request.request_id}")

                    # Send with consistent framing
                    self.dealer_socket.send_multipart([b"", pickle.dumps(request)])
                    self._debug(
                        f"Successfully sent queued request {request.request_id}"
                    )

                except Empty:
                    continue
                except Exception as e:
                    self._error(f"Error in send loop: {str(e)}")
                    self._debug(f"Traceback: {traceback.format_exc()}")
                    continue

    def _handle_responses(self):
        """Handle incoming responses with detailed debugging."""
        self._debug("Response handler thread started")
        while self.running:
            try:
                response = self.receive_response(timeout=100)  # in ms
                if response is None:
                    continue

                request_id = response.request_id
                self._debug(f"Received response for request_id: {request_id}")

                with self.pending_responses_lock:
                    if request_id in self.pending_responses:
                        self._debug(f"Found pending queue for request_id: {request_id}")
                        queue = self.pending_responses[request_id]
                        queue.put(response.data)
                        self._debug(
                            f"Response data added to queue for request_id: {request_id}"
                        )
                    else:
                        self._error(
                            f"No pending queue found for request_id: {request_id}"
                        )
                        self._debug(
                            f"Current pending requests: {list(self.pending_responses.keys())}"
                        )

                with self.active_requests_lock:
                    if request_id in self.active_requests:
                        self.active_requests.remove(request_id)
                        self._debug(
                            f"Request {request_id} removed from active requests"
                        )

            except Exception as e:
                self._error(f"Error in response handler: {str(e)}")
                self._debug(f"Traceback: {traceback.format_exc()}")
                continue

    def receive_data(self, timeout: Optional[int] = None) -> Optional[dict]:
        """Receive data from the subscription stream."""
        if timeout is not None:
            if self.sub_socket.poll(timeout) == 0:
                return None

        _, pickled_msg = self.sub_socket.recv_multipart()
        return pickle.loads(pickled_msg)

    def send_request(self, request: Request, timeout=1.0):
        """Send request with queuing and timeout."""
        if not isinstance(request, Request):
            raise ValueError("Must send a Request object")

        self._debug(f"Sending request: {request.type} with ID: {request.request_id}")

        # Create a queue for this specific request
        response_queue = Queue()

        with self.pending_responses_lock:
            self._debug(f"Adding request {request.request_id} to pending_responses")
            self.pending_responses[request.request_id] = response_queue

        with self.active_requests_lock:
            self.active_requests.add(request.request_id)
            self._debug(f"Added {request.request_id} to active requests")

        try:
            self._debug(f"Queueing request {request.request_id}")
            self.send_queue.put(request)

            self._debug(f"Waiting for response with timeout {timeout}")
            try:
                response = response_queue.get(timeout=timeout)
                self._debug(f"Received response for request {request.request_id}")
                return response
            except Empty:
                self._error(
                    f"Timeout waiting for response to request {request.request_id}"
                )
                self._debug(f"Active requests at timeout: {self.active_requests}")
                raise TimeoutError(f"Request {request.type} timed out")

        finally:
            with self.pending_responses_lock:
                if request.request_id in self.pending_responses:
                    self._debug(
                        f"Cleaning up request {request.request_id} from pending_responses"
                    )
                    del self.pending_responses[request.request_id]
                else:
                    self._debug(
                        f"Request {request.request_id} not found in pending_responses during cleanup"
                    )

    def receive_response(self, timeout=None):
        """Non-blocking receive for responses with debugging."""
        try:
            if self.dealer_socket.poll(
                timeout=timeout if timeout else 0, flags=zmq.POLLIN
            ):
                message_parts = self.dealer_socket.recv_multipart()
                if len(message_parts) != 2:  # Expecting [delimiter, response]
                    self._error(
                        f"Received {len(message_parts)} message parts, expected 2"
                    )
                    return None

                empty, response = message_parts
                if empty != b"":
                    self._error("Invalid delimiter frame")
                    return None

                response_obj = pickle.loads(response)
                self._debug(
                    f"Received raw response for request: {response_obj.request_id}"
                )
                return response_obj

        except Exception as e:
            self._error(f"Error in receive_response: {str(e)}")
        return None

    def close(self):
        """Clean up with debugging."""
        self._debug("Starting client shutdown")
        self.running = False

        # Clear pending responses
        with self.pending_responses_lock:
            self._debug(f"Clearing {len(self.pending_responses)} pending responses")
            self.pending_responses.clear()

        # Wait for threads to finish
        if hasattr(self, "response_thread"):
            self._debug("Joining response thread")
            self.response_thread.join(timeout=2.0)

        if hasattr(self, "send_thread"):
            self._debug("Joining send thread")
            self.send_thread.join(timeout=2.0)

        # Close sockets
        self._debug("Closing sockets")
        self.sub_socket.close()
        self.dealer_socket.close()
        self.sub_context.term()
        self.dealer_context.term()
        self._debug("Client shutdown complete")
