import os
import time
from dataclasses import dataclass, fields
from queue import Empty, Full, Queue  # Explicitly import exceptions
from threading import Event, Lock, Thread
from typing import List, Optional, Type

import cv2
import imageio
import numpy as np
import zarr

from dexumi.common.frame_manager import FrameRateContext
from dexumi.common.imagecodecs_numcodecs import JpegXl, register_codecs
from dexumi.encoder.numeric import Numeric, NumericFrame

register_codecs()


class NumericRecorder:
    def __init__(
        self,
        record_fps: int,
        stream_fps: int,
        record_path: str,
        numeric_sources: List[Numeric],
        frame_data_class: NumericFrame,
        verbose=False,
        frame_queue_size: int = 20,
    ):
        self.record_fps = record_fps
        self.record_path = record_path
        self.numeric_sources = numeric_sources
        self.frame_data_class = frame_data_class
        self.verbose = verbose
        self.frame_queues: List[Queue] = [
            Queue(maxsize=frame_queue_size) for _ in numeric_sources
        ]
        self._streaming_event: Event = Event()
        self._recording_event: Event = Event()
        self.record_dt: float = 1 / record_fps

        # Create directory if it doesn't exist
        os.makedirs(record_path, exist_ok=True)
        self.record_roots = zarr.open(record_path, mode="a")
        self.stream_fps: int = stream_fps

        # Initialize thread lists
        self.numeric_threads: List[Thread] = []
        self.recording_threads: List[Thread] = []

        self._episode_id = self.init_episode_id()
        print(f"Starting episode {self.episode_id}")

    @property
    def episode_id(self):
        return self._episode_id

    @property
    def is_recording(self) -> bool:
        """Check if currently recording numeric data.

        Returns:
            bool: True if recording is in progress, False otherwise.
        """
        return self._recording_event.is_set()

    def init_episode_id(self):
        keys = list(self.record_roots.group_keys())
        return len(keys)

    def set_episode_id(self, episode_id: int = None) -> None:
        self._episode_id = episode_id

    def reset_episode_recording(self):
        if self._recording_event.is_set():
            """
            call this function when recording already start will cause no data to be logged.
            the reference passed to the record thread will be replaced.
            """
            return False

        os.makedirs(
            os.path.join(self.record_path, f"episode_{self.episode_id}"),
            exist_ok=True,
        )

        # Initialize empty list for each numeric source
        self.episode_frames: List[dict] = [{} for _ in self.numeric_sources]

        # Get frame data from each source to initialize fields
        for i in range(len(self.numeric_sources)):
            # Try to get latest frame data
            frame_data = self.peek_latest_frame(i)
            if frame_data is None:
                # If no frame available, fall back to frame_data_class fields
                self.episode_frames[i] = {
                    field.name: [] for field in fields(self.frame_data_class)
                }
            else:
                # Use actual frame data fields
                self.episode_frames[i] = {
                    field: [] for field in vars(frame_data).keys()
                }

        return True

    def clear_recording(self):
        """Delete the current episode's folder and reset episode ID."""
        if self._recording_event.is_set():
            print("Cannot delete while recording is in progress.")
            return False

        episode_path = os.path.join(self.record_path, f"episode_{self.episode_id}")
        try:
            if os.path.exists(episode_path):
                # Delete all files in the episode directory
                for file in os.listdir(episode_path):
                    file_path = os.path.join(episode_path, file)
                    os.remove(file_path)
                # Remove the directory itself
                os.rmdir(episode_path)
                print(f"Deleted episode {self.episode_id}")
                return True
        except Exception as e:
            print(f"Error deleting episode: {e}")
        return False

    def _numeric_stream_thread(self, numeric_idx):
        numeric = self.numeric_sources[numeric_idx]
        queue = self.frame_queues[numeric_idx]
        frame_received = 0
        start_time = time.monotonic()

        while self._streaming_event.is_set():
            with FrameRateContext(self.stream_fps, verbose=self.verbose) as fr:
                try:
                    frame_data = numeric.get_numeric_frame()
                    if frame_data is not None:
                        frame_received += 1
                        if self.verbose:
                            actual_fps = frame_received / (
                                time.monotonic() - start_time
                            )
                            print(
                                f"Streaming Numeric {numeric_idx} FPS: {actual_fps:.2f}"
                            )
                        if queue.full():
                            try:
                                queue.get_nowait()
                            except Empty:
                                pass
                        queue.put_nowait(frame_data)
                except Exception as e:
                    print(f"Error in numeric stream thread: {e}")
                    time.sleep(0.1)

    def _recording_thread(self, numeric_idx, episode_id=None):
        stored_frame = self.episode_frames[numeric_idx]
        frame_received = 0
        start_time = time.monotonic()
        last_timestamp = None
        while self._recording_event.is_set():
            with FrameRateContext(self.record_fps, verbose=self.verbose) as fr:
                frame_data = self.get_next_frame(
                    numeric_idx, last_timestamp, timeout=1 / self.record_fps
                )
                if frame_data is None:
                    print(f"No frames in queue for numeric_{numeric_idx}")
                    continue
                last_timestamp = frame_data.receive_time
                frame_received += 1
                if self.verbose:
                    actual_fps = frame_received / (time.monotonic() - start_time)
                    print(f"Recording numeric_{numeric_idx} FPS: {actual_fps:.2f}")

                # Save all available fields from frame_data
                for field_name, value in vars(frame_data).items():
                    if field_name in stored_frame:
                        if value is not None:
                            stored_frame[field_name].append(value)
                    else:
                        print(f"Warning: Field {field_name} not found in stored_frame")

    def peek_latest_frame(self, numeric_idx):
        if 0 <= numeric_idx < len(self.numeric_sources):
            queue = self.frame_queues[numeric_idx]
            with queue.mutex:
                if queue.queue:
                    return queue.queue[-1]
        return None

    def get_last_k_frames(
        self, numeric_idx: int, k: int, remove: bool = False
    ) -> Optional[List[NumericFrame]]:
        """
        Retrieve the last k frames from the queue for a given numeric source,
        with option to remove them from the queue.

        Args:
            numeric_idx (int): Index of the numeric source.
            k (int): Number of frames to retrieve.
            remove (bool): Whether to remove the retrieved frames from queue.

        Returns:
            List: The last k frames from the queue, or None if the queue is empty.
        """
        if 0 <= numeric_idx < len(self.numeric_sources):
            queue = self.frame_queues[numeric_idx]
            with queue.mutex:
                if queue.queue:
                    queue_list = list(queue.queue)
                    frames = queue_list[-k:]
                    if remove:
                        # Clear the retrieved frames from queue
                        for _ in range(min(k, len(queue_list))):
                            queue.queue.pop()
                    return frames
        return None

    def get_last_k_frames_from_all(self, k: int, remove: bool = False) -> dict:
        """
        Retrieve the last k frames from the queue for all numeric sources,
        with option to remove them from the queue.

        Args:
            k (int): Number of frames to retrieve from each numeric source.
            remove (bool): Whether to remove the retrieved frames from queues.

        Returns:
            dict: A dictionary where keys are device names and values are lists of the last k frames.
        """
        frames_dict = {}
        for i, device in enumerate(self.numeric_sources):
            device_name = getattr(device, "device_name", f"numeric_{i}")
            frames = self.get_last_k_frames(i, k, remove)
            if frames is not None:
                frames_dict[device_name] = frames
            else:
                frames_dict[device_name] = None
        return frames_dict

    def get_next_frame(
        self,
        numeric_idx: int,
        last_timestamp: Optional[float] = None,
        timeout: float = 1.0,
    ) -> Optional[NumericFrame]:
        """Get the newest numeric frame that has a receive_time larger than the last timestamp,
        waiting if necessary until a newer frame arrives.

        Args:
            numeric_idx: Index of the numeric source
            last_timestamp: The timestamp of the last frame that was processed.
                        If None, will return the newest frame in queue.
            timeout: Maximum time to wait for a newer frame in seconds

        Returns:
            Optional[NumericFrame]: The newest frame with receive_time > last_timestamp,
                                or None if no such frame exists within timeout period
        """
        if not 0 <= numeric_idx < len(self.numeric_sources):
            return None

        queue = self.frame_queues[numeric_idx]
        start_time = time.monotonic()

        while (time.monotonic() - start_time) < timeout:
            with queue.mutex:
                if not queue.queue:
                    continue

                newest_frame = queue.queue[-1]
                if last_timestamp is None or (
                    newest_frame.receive_time is not None
                    and newest_frame.receive_time > last_timestamp
                ):
                    return newest_frame

            time.sleep(0.001)

        return None

    def start_streaming(self):
        if self._streaming_event.is_set():
            print("Streaming is already in progress.")
            return False

        self._streaming_event.set()

        print("Starting numeric streaming...")
        for numeric in self.numeric_sources:
            numeric.start_streaming()

        time.sleep(0.1)

        for i in range(len(self.numeric_sources)):
            thread = Thread(target=self._numeric_stream_thread, args=(i,))
            thread.daemon = True
            thread.start()
            self.numeric_threads.append(thread)

        return True

    def start_recording(self, episode_id=None):
        if episode_id is not None:
            self.set_episode_id(episode_id)
        if self._recording_event.is_set():
            print("Recording is already in progress.")
            return False

        self._recording_event.set()
        time.sleep(0.5)

        try:
            print("Starting recording threads...")
            self.recording_threads = []
            for i in range(len(self.numeric_sources)):
                thread = Thread(
                    target=self._recording_thread, args=(i, self.episode_id)
                )
                thread.daemon = True
                thread.start()
                self.recording_threads.append(thread)

            print("Recording started successfully.")
            return True

        except Exception as e:
            print(f"Failed to start recording: {e}")
            self.stop_recording()
            return False

    def stop_streaming(self):
        if not self._streaming_event.is_set():
            print("Streaming is not in progress.")
            return False

        print("Stopping streaming...")
        self._streaming_event.clear()

        print("Waiting for threads to finish...")
        for thread in self.numeric_threads:
            thread.join(timeout=2.0)

        for numeric in self.numeric_sources:
            try:
                numeric.stop_streaming()
            except Exception as e:
                print(f"Error stopping numeric source: {e}")

        print("Streaming stopped successfully.")

        return True

    def stop_recording(self):
        if not self._recording_event.is_set():
            print("Recording is not in progress.")
            return False

        print("Stopping recording...")
        self._recording_event.clear()

        print("Waiting for threads to finish...")
        for thread in self.recording_threads:
            thread.join(timeout=2.0)

        return True

    def save_recordings(self):
        this_episode_id = self._episode_id
        episode_data = self.record_roots.require_group(f"episode_{this_episode_id}")

        for i in range(len(self.numeric_sources)):
            print(f"Writing numeric_{i} data to disk...")
            episode_numeric_data = episode_data.require_group(f"numeric_{i}")
            episode_to_save = self.episode_frames[i]

            # Save all fields that have data
            for field_name, value_list in episode_to_save.items():
                if value_list:  # Only save non-empty lists
                    print(f"Writing {field_name} data...")
                    # Convert to numpy array for zarr 3.x compatibility
                    episode_numeric_data[field_name] = np.array(value_list)
                else:
                    print(f"No {field_name} data to save for numeric_{i}")

        self.set_episode_id(this_episode_id + 1)
        print("Recording saved successfully.")


if __name__ == "__main__":
    from dexumi.encoder.encoder import InspireEncoder, JointFrame

    numeric_recorder = NumericRecorder(
        record_fps=30,
        record_path=os.path.expanduser("~/Dev/DexUMI/data_local/numeric"),
        numeric_sources=[InspireEncoder("inspire", verbose=False)],
        frame_data_class=JointFrame,
        verbose=False,
    )

    numeric_recorder.start_streaming()

    while True:
        frame_data = numeric_recorder.peek_latest_frame(0)
        if frame_data is not None:
            # generate a random rgb
            viz_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.putText(
                viz_frame,
                f"Episode: {numeric_recorder.episode_id}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("RGB", viz_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                numeric_recorder.stop_recording()
                numeric_recorder.stop_streaming()
                break
            elif key == ord("s"):
                if numeric_recorder.reset_episode_recording():
                    print("Starting recording...")
                    numeric_recorder.start_recording()
                else:
                    print("Recording already started.")
            elif key == ord("w"):
                print("Saving recording...")
                if numeric_recorder.stop_recording():
                    numeric_recorder.save_recordings()
            elif key == ord("a"):
                print("Restarting episode...")
                if numeric_recorder.stop_recording():
                    numeric_recorder.clear_recording()

        time.sleep(1 / 30)  # Approximate 30 FPS
