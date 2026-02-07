import copy
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, fields
from queue import Empty, Full, Queue  # Explicitly import exceptions
from threading import Event, Lock, Thread
from typing import List, Optional, Type

import cv2
import imageio
import numpy as np
import zarr

from dexumi.camera.camera import Camera, FrameData, FrameNumericData

# from dexumi.camera.iphone_camera import IphoneCamera
# OakCamera import moved to __main__ block to avoid depthai dependency
from dexumi.common.frame_manager import FrameRateContext
from dexumi.common.imagecodecs_numcodecs import JpegXl, register_codecs
from dexumi.common.utility.zarr import parallel_saving

register_codecs()


class VideoRecorder:
    def __init__(
        self,
        record_fps: int,
        stream_fps: int,
        video_record_path: str,
        camera_sources: List[Camera],
        frame_data_class: Type[FrameData],
        verbose: bool = False,
        convert_bgr_to_rgb: bool = False,
        max_workers: int = 32,
        frame_queue_size: int = 20,
    ) -> None:
        self.video_record_path: str = video_record_path
        self.camera_sources: List[Camera] = camera_sources
        camera_names = [camera.camera_name for camera in camera_sources]
        if np.unique(camera_names).size != len(camera_names):
            raise ValueError("Camera names must be unique.")
        self.frame_data_class: Type[FrameNumericData] = frame_data_class
        self.verbose: bool = verbose
        self._streaming_event: Event = Event()
        self._recording_event: Event = Event()
        self.frame_queues: List[Queue] = [
            Queue(maxsize=frame_queue_size) for _ in camera_sources
        ]
        self.record_fps: int = record_fps
        self.stream_fps: int = stream_fps
        self.record_dt: float = 1 / record_fps
        self.convert_bgr_to_rgb: bool = convert_bgr_to_rgb
        # Create directory if it doesn't exist
        os.makedirs(video_record_path, exist_ok=True)
        self.record_roots = zarr.open(video_record_path, mode="a")
        self.max_workers = max_workers

        # Initialize thread lists
        self.camera_threads: List[Thread] = []
        self.recording_threads: List[Thread] = []
        self.writer_threads: List[Thread] = []
        # Buffered frame queues for async video writing (deques per camera)
        self.write_buffers: List[deque] = [deque() for _ in camera_sources]
        self._writing_event: Event = Event()  # Signals writer threads to keep running

        self._episode_id = self.init_episode_id()
        print(f"Starting episode {self.episode_id}")

    @property
    def episode_id(self):
        return self._episode_id

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
            os.path.join(self.video_record_path, f"episode_{self.episode_id}"),
            exist_ok=True,
        )
        # Delete existing video files if they exist
        for i in range(len(self.camera_sources)):
            video_path = os.path.join(
                self.video_record_path, f"episode_{self.episode_id}", f"camera_{i}.mp4"
            )
            if os.path.exists(video_path):
                os.remove(video_path)

        self.video_writers: List[Optional[cv2.VideoWriter]] = [
            None for _ in self.camera_sources
        ]
        self.episode_frames: List[dict] = [{} for _ in self.camera_sources]
        for i in range(len(self.camera_sources)):
            self.episode_frames[i] = {
                field.name: [] for field in fields(self.frame_data_class)
            }

        return True

    def clear_recording(self):
        """Delete the current episode's folder and reset episode ID."""
        if self._recording_event.is_set():
            print("Cannot delete while recording is in progress.")
            return False

        episode_path = os.path.join(
            self.video_record_path, f"episode_{self.episode_id}"
        )
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

    def _camera_stream_thread(self, camera_idx):
        """Thread function to continuously get frames from a camera"""
        camera = self.camera_sources[camera_idx]
        queue = self.frame_queues[camera_idx]
        start_time = time.monotonic()
        frame_received = 0
        # Debug timing
        debug_interval = 60  # Print every N frames
        t_get_total = 0.0
        t_queue_total = 0.0
        while self._streaming_event.is_set():
            try:
                with FrameRateContext(self.stream_fps, verbose=self.verbose) as fr:
                    t0 = time.monotonic()
                    frame_data = camera.get_camera_frame()
                    t1 = time.monotonic()
                    if frame_data.rgb is not None:
                        # Check if queue is full first
                        frame_received += 1
                        t_get_total += (t1 - t0)
                        if queue.full():
                            try:
                                queue.get_nowait()  # Remove oldest frame
                            except Empty:
                                pass

                        try:
                            queue.put_nowait(frame_data)
                        except Full:
                            pass
                        t2 = time.monotonic()
                        t_queue_total += (t2 - t1)

                        if frame_received % debug_interval == 0:
                            elapsed = time.monotonic() - start_time
                            avg_fps = frame_received / elapsed
                            print(
                                f"[STREAM cam{camera_idx}] fps={avg_fps:.1f} "
                                f"get_frame={1000*t_get_total/debug_interval:.1f}ms "
                                f"queue={1000*t_queue_total/debug_interval:.2f}ms "
                                f"qsize={queue.qsize()}"
                            )
                            t_get_total = 0.0
                            t_queue_total = 0.0
                    else:
                        print(f"No frame data from camera {camera_idx}")
            except Exception as e:
                print(f"Error capturing frame from camera {camera_idx}: {e}")
                time.sleep(0.1)

    def _recording_thread(self, camera_idx, episode_id):
        """Thread function to capture frames from queue into memory buffer.
        Video encoding happens in a separate writer thread to avoid blocking."""
        stored_frame = self.episode_frames[camera_idx]
        video_path = os.path.join(
            self.video_record_path,
            f"episode_{episode_id}",
            f"camera_{camera_idx}.mp4",
        )
        fps = self.record_fps
        # Get first frame to determine dimensions and initialize video writer
        first_frame = self.peek_latest_frame(camera_idx)
        while first_frame is None:
            print("Waiting for first frame to initialize video writer...")
            first_frame = self.peek_latest_frame(camera_idx)

        frame = first_frame.rgb
        height, width = frame.shape[:2]
        print("video_path", video_path)
        self.video_writers[camera_idx] = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if self.verbose:
            print(f"Initialized video writer for camera {camera_idx}")

        # Clear the write buffer and start the async writer thread
        self.write_buffers[camera_idx].clear()

        try:
            last_timestamp = first_frame.receive_time
            start_time = time.monotonic()
            frame_received = 0
            missed_frames = 0
            debug_interval = 60

            while self._recording_event.is_set():
                with FrameRateContext(fps, verbose=self.verbose) as fr:
                    frame_data = self.get_next_frame(
                        camera_idx, last_timestamp, timeout=1 / fps
                    )
                    if frame_data is None:
                        missed_frames += 1
                        continue

                    frame_received += 1
                    last_timestamp = frame_data.receive_time

                    for field in fields(self.frame_data_class):
                        value = getattr(frame_data, field.name)
                        if value is not None:
                            if field.name == "rgb":
                                if self.convert_bgr_to_rgb:
                                    value = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
                                # Buffer frame for async writing instead of blocking
                                self.write_buffers[camera_idx].append(value)
                            if field.name in self.frame_data_class.numeric_fields():
                                stored_frame[field.name].append(value)

                    if frame_received % debug_interval == 0:
                        elapsed = time.monotonic() - start_time
                        avg_fps = frame_received / elapsed
                        buf_len = len(self.write_buffers[camera_idx])
                        print(
                            f"[REC cam{camera_idx}] capture_fps={avg_fps:.1f} "
                            f"buf={buf_len} missed={missed_frames}"
                        )
        except Exception as e:
            import traceback
            print(f"Error in recording thread camera {camera_idx}: {e}")
            traceback.print_exc()

    def _video_writer_thread(self, camera_idx):
        """Async thread that drains the write buffer to cv2.VideoWriter.
        Runs during recording and continues after stop until buffer is empty."""
        video_writer = self.video_writers[camera_idx]
        write_buf = self.write_buffers[camera_idx]
        frames_written = 0
        start_time = time.monotonic()

        while self._writing_event.is_set() or len(write_buf) > 0:
            if len(write_buf) > 0:
                frame = write_buf.popleft()
                video_writer.write(frame)
                frames_written += 1
                if frames_written % 30 == 0:
                    elapsed = time.monotonic() - start_time
                    print(
                        f"[WRITER cam{camera_idx}] encode_fps={frames_written/elapsed:.1f} "
                        f"written={frames_written} buf={len(write_buf)}"
                    )
            else:
                time.sleep(0.001)

        print(f"[WRITER cam{camera_idx}] done. Total frames written: {frames_written}")

    def peek_latest_frame(self, camera_idx: int) -> Optional[FrameData]:
        """Peek the latest frame from the queue without removing it"""
        if 0 <= camera_idx < len(self.frame_queues):
            queue = self.frame_queues[camera_idx]
            # print("queue size", queue.qsize())
            with queue.mutex:
                if queue.queue:
                    # print(f"Peeked frame from camera {camera_idx}")
                    frame = queue.queue[-1]
                    # if self.verbose:
                    #     print(f"Peeked frame from camera {camera_idx}")
                    return frame
            # print("No frames in queue.")
        return None

    def get_last_k_frames(
        self, camera_idx: int, k: int, remove: bool = False
    ) -> List[FrameData]:
        """
        Get the latest k frames from the queue with option to remove them.

        Args:
            camera_idx: Camera index
            k: Number of frames to get
            remove: Whether to remove the frames from queue
        """
        if 0 <= camera_idx < len(self.frame_queues):
            queue = self.frame_queues[camera_idx]
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
        """Get the latest k frames from all cameras.

        Args:
            k (int): Number of latest frames to retrieve

        Returns:
            dict: Dictionary with camera names as keys and lists of frames as values
        """
        frames_dict = {}
        for i, camera in enumerate(self.camera_sources):
            camera_name = getattr(camera, "camera_name", f"camera_{i}")
            frames = self.get_last_k_frames(i, k, remove)
            if frames is not None:
                frames_dict[camera_name] = frames
            else:
                frames_dict[camera_name] = None

        return frames_dict

    def get_next_frame(
        self,
        camera_idx: int,
        last_timestamp: Optional[float] = None,
        timeout: float = 1.0,
    ) -> Optional[FrameData]:
        """Get the newest frame from the queue that has a receive_time larger than the last timestamp,
        waiting if necessary until a newer frame arrives.

        Args:
            camera_idx: Index of the camera to get frame from
            last_timestamp: The timestamp of the last frame that was processed.
                        If None, will return the newest frame in queue.
            timeout: Maximum time to wait for a newer frame in seconds

        Returns:
            Optional[FrameData]: The newest frame with receive_time > last_timestamp, or None if no such frame exists
        """
        if not 0 <= camera_idx < len(self.frame_queues):
            return None

        queue = self.frame_queues[camera_idx]
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

        # Set streaming event
        self._streaming_event.set()
        # Start camera streams
        print("Starting camera streams...")
        for camera in self.camera_sources:
            if hasattr(camera, "start_streaming"):
                camera.start_streaming()

        # Give cameras time to initialize
        time.sleep(0.5)

        # Then start camera streaming threads
        print("Starting camera threads...")
        self.camera_threads = []
        for i in range(len(self.camera_sources)):
            thread = Thread(target=self._camera_stream_thread, args=(i,))
            thread.daemon = True
            thread.start()
            self.camera_threads.append(thread)

        return True

    def start_recording(self, episode_id=None):
        """Start recording from all cameras"""
        if episode_id is not None:
            self.set_episode_id(episode_id)
        if self._recording_event.is_set():
            print("Recording is already in progress.")
            return False
        self._recording_event.set()
        self._writing_event.set()
        time.sleep(0.5)
        try:
            print("Starting recording threads...")
            self.recording_threads = []
            self.writer_threads = []
            for i in range(len(self.camera_sources)):
                thread = Thread(
                    target=self._recording_thread, args=(i, self.episode_id)
                )
                thread.daemon = True
                thread.start()
                self.recording_threads.append(thread)

            # Writer threads start after recording threads (need video_writers initialized)
            time.sleep(0.5)
            for i in range(len(self.camera_sources)):
                wthread = Thread(
                    target=self._video_writer_thread, args=(i,)
                )
                wthread.daemon = True
                wthread.start()
                self.writer_threads.append(wthread)

            print("Recording started successfully.")
            return True

        except Exception as e:
            print(f"Failed to start recording: {e}")
            self.stop_recording()
            return False

    def stop_streaming(self):
        """Stop streaming from all cameras"""
        if not self._streaming_event.is_set():
            print("Streaming is not in progress.")
            return False

        print("Stopping streaming...")
        self._streaming_event.clear()

        # Wait for threads to finish
        print("Waiting for threads to finish...")
        for thread in self.camera_threads:
            thread.join(timeout=2.0)

        # Stop camera streams
        print("Stopping camera streams...")
        for camera in self.camera_sources:
            try:
                if hasattr(camera, "stop_streaming"):
                    camera.stop_streaming()
            except Exception as e:
                print(f"Error stopping camera: {e}")

        print("Streaming stopped successfully.")

        return True

    def stop_recording(self):
        """Stop recording and save all data"""
        if not self._recording_event.is_set():
            print("Recording is not in progress.")
            return False

        print("Stopping recording...")
        self._recording_event.clear()

        # Wait for recording threads to finish
        print("Waiting for recording threads to finish...")
        for thread in self.recording_threads:
            thread.join(timeout=2.0)

        # Signal writer threads to stop (they will drain remaining buffer first)
        self._writing_event.clear()
        total_remaining = sum(len(b) for b in self.write_buffers)
        if total_remaining > 0:
            print(f"Encoding {total_remaining} buffered frames to video...")
        for thread in self.writer_threads:
            thread.join(timeout=120.0)  # Allow up to 2 min to drain buffer

        return True

    def save_recordings(self):
        """Save all recorded frames to disk"""
        this_episode_id = self._episode_id
        episode_data = self.record_roots.require_group(f"episode_{this_episode_id}")

        # Release all video writers
        print("Releasing video writers...")
        for i, writer in enumerate(self.video_writers):
            if writer is not None:
                writer.release()
                # self.video_writers[i] = None
        for i in range(len(self.camera_sources)):
            if self.verbose:
                print(f"Writing camera_{i} data to disk...")
            episode_camera_data = episode_data.require_group(f"camera_{i}")
            episode_to_save = self.episode_frames[i]

            for key, value_list in episode_to_save.items():
                if not value_list:
                    if self.verbose:
                        print(f"No {key} data to save for camera_{i}")
                    continue
                if self.verbose:
                    print(f"Writing {key} data...")
                # Convert to numpy array for zarr 3.x compatibility
                episode_camera_data[key] = np.array(value_list)

        self.set_episode_id(this_episode_id + 1)
        print("Recording saved successfully.")

    @property
    def is_recording(self) -> bool:
        """Check if currently recording video.

        Returns:
            bool: True if recording is in progress, False otherwise.
        """
        return self._recording_event.is_set()


if __name__ == "__main__":
    import time
    from dexumi.camera.oak_camera import OakCamera, get_all_oak_cameras

    # Test code
    all_cam = get_all_oak_cameras()
    cam_0 = OakCamera("oak_0", device_id=all_cam[0])
    cam_1 = OakCamera("oak_1", device_id=all_cam[1])
    video_recorder = VideoRecorder(
        record_fps=30,
        video_record_path=os.path.expanduser("~/Dev/DexUMI/data_local/video"),
        frame_data_class=FrameData,
        camera_sources=[cam_0, cam_1],
        # camera_sources=[cam_0],
        verbose=False,
        convert_bgr_to_rgb=False,
    )
    fps = 30
    dt = 1 / fps
    video_recorder.start_streaming()
    start_time = time.monotonic()
    iter = 0
    while True:
        end_time = start_time + (1 + iter) * dt
        frames_dict = video_recorder.get_lask_k_frames_from_all_cameras(k=1)

        # Show frames from all cameras
        frames_to_show = []
        for camera_name, frames in frames_dict.items():
            if frames and frames[0] is not None:
                # Get first frame (k=1)
                frame_data = frames[0]
                # Add episode number overlay
                viz_frame = copy.deepcopy(frame_data.rgb)
                cv2.putText(
                    viz_frame,
                    f"Episode: {video_recorder.episode_id} - {camera_name}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                frames_to_show.append(viz_frame)

        if frames_to_show:
            # Calculate dimensions
            n_frames = len(frames_to_show)

            # Ensure all frames have same dimensions by resizing
            h, w = frames_to_show[0].shape[:2]
            frames_to_show = [cv2.resize(frame, (w, h)) for frame in frames_to_show]

            # Concatenate horizontally
            combined_frame = np.hstack(frames_to_show)

            # Show combined frame
            cv2.imshow("All Cameras", combined_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            if video_recorder.stop_recording():
                video_recorder.clear_recording()
            video_recorder.stop_streaming()
            break
        elif key == ord("s"):
            if video_recorder.reset_episode_recording():
                print("Starting recording...")
                video_recorder.start_recording()
            else:
                print("Recording already started.")
        elif key == ord("w"):
            print("Saving recording...")
            if video_recorder.stop_recording():
                video_recorder.save_recordings()
        elif key == ord("a"):
            print("Restarting episode...")
            if video_recorder.stop_recording():
                video_recorder.clear_recording()

        if time.monotonic() < end_time:
            time.sleep(end_time - time.monotonic())
        iter += 1
