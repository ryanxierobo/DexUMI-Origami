import threading
import time

import numpy as np
from spnav import (
    SpnavButtonEvent,
    SpnavMotionEvent,
    spnav_close,
    spnav_open,
    spnav_poll_event,
)

from dexumi.real_env.ring_buffer import RingBuffer


class Spacemouse:
    def __init__(
        self,
        get_max_k=30,
        frequency=200,
        max_value=500,
        deadzone=(0, 0, 0, 0, 0, 0),
        dtype=np.float32,
        n_buttons=2,
    ):
        """
        Continuously listen to 3D connection space navigator events
        and update the latest state.

        max_value: {300, 500} 300 for wired version and 500 for wireless
        deadzone: [0,1], number or tuple, axis with value lower than this value will stay at 0

        front
        z
        ^   _
        |  (O) space mouse
        |
        *----->x right
        y
        """
        if np.issubdtype(type(deadzone), np.number):
            deadzone = np.full(6, fill_value=deadzone, dtype=dtype)
        else:
            deadzone = np.array(deadzone, dtype=dtype)
        assert (deadzone >= 0).all()

        # instance variables
        self.frequency = frequency
        self.max_value = max_value
        self.dtype = dtype
        self.deadzone = deadzone
        self.n_buttons = n_buttons
        self.tx_zup_spnav = np.array([[0, 0, -1], [1, 0, 0], [0, 1, 0]], dtype=dtype)

        # State variables
        self.motion_event = np.zeros((7,), dtype=np.int64)
        self.button_state = np.zeros((self.n_buttons,), dtype=bool)
        self.receive_timestamp = time.time()

        # Thread control
        self.ready_event = threading.Event()
        self.stop_event = threading.Event()
        self.thread = None

        # Buffer for history if needed
        self.ring_buffer = RingBuffer(get_max_k)

        # Initialize connection
        self.is_connected = False

    # ======= get state APIs ==========

    def get_motion_state(self):
        """Get the current motion state normalized by max_value and apply deadzone"""
        state = np.array(self.motion_event[:6], dtype=self.dtype) / self.max_value
        is_dead = (-self.deadzone < state) & (state < self.deadzone)
        state[is_dead] = 0
        return state

    def get_motion_state_transformed(self):
        """
        Return in right-handed coordinate
        z
        *------>y right
        |   _
        |  (O) space mouse
        v
        x
        back
        """
        state = self.get_motion_state()
        tf_state = np.zeros_like(state)
        tf_state[:3] = self.tx_zup_spnav @ state[:3]
        tf_state[3:] = self.tx_zup_spnav @ state[3:]
        return tf_state

    def get_button_state(self):
        """Get the current button state"""
        return self.button_state

    def is_button_pressed(self, button_id):
        """Check if a specific button is pressed"""
        if 0 <= button_id < self.n_buttons:
            return self.button_state[button_id]
        return False

    # ========== start stop API ===========

    def start(self, wait=True):
        """Start the spacemouse listener thread"""
        if self.thread is not None and self.thread.is_alive():
            return  # Already running

        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

        if wait:
            self.ready_event.wait()

    def stop(self, wait=True):
        """Stop the spacemouse listener thread"""
        if self.thread is None or not self.thread.is_alive():
            return  # Already stopped

        self.stop_event.set()

        if wait and self.thread is not None:
            self.thread.join()
            self.thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= main loop ==========
    def _run(self):
        """Main loop for polling spacemouse events"""
        try:
            spnav_open()
            # Initialize state
            self.motion_event = np.zeros((7,), dtype=np.int64)
            self.button_state = np.zeros((self.n_buttons,), dtype=bool)
            self.receive_timestamp = time.time()

            # Store initial state in ring buffer
            self.ring_buffer.write(
                {
                    "motion_event": np.copy(self.motion_event),
                    "button_state": np.copy(self.button_state),
                    "receive_timestamp": self.receive_timestamp,
                }
            )

            # Signal that we're ready
            self.ready_event.set()
            print("spacemouse ready")

            while not self.stop_event.is_set():
                event = spnav_poll_event()
                self.receive_timestamp = time.time()

                if isinstance(event, SpnavMotionEvent):
                    self.motion_event[:3] = event.translation
                    self.motion_event[3:6] = event.rotation
                    self.motion_event[6] = event.period

                elif isinstance(event, SpnavButtonEvent):
                    if 0 <= event.bnum < self.n_buttons:
                        self.button_state[event.bnum] = event.press

                # Store current state in ring buffer
                self.ring_buffer.write(
                    {
                        "motion_event": np.copy(self.motion_event),
                        "button_state": np.copy(self.button_state),
                        "receive_timestamp": self.receive_timestamp,
                    }
                )

                # Sleep to maintain desired frequency
                time.sleep(1 / self.frequency)

        finally:
            if self.is_connected:
                spnav_close()
                self.is_connected = False
