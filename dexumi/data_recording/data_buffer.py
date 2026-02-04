import logging
import os
import shutil
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import zarr
from dexumi.common.imagecodecs_numcodecs import JpegXl, register_codecs
from dexumi.common.utility.matrix import (
    convert_homogeneous_matrix,
)
from dexumi.common.utility.video import extract_frames_videos
from scipy import interpolate
from scipy.spatial import transform as st
from scipy.spatial.transform import Rotation

# Configure logger
logger = logging.getLogger(__name__)


class PoseInterpolator:
    def __init__(self, timestamps, homogeneous_matrix):
        positions = homogeneous_matrix[:, :3, 3]
        rotations = Rotation.from_matrix(homogeneous_matrix[:, :3, :3])

        # Find indices where timestamps are not monotonically increasing
        drop_indices = [
            idx
            for idx in range(1, len(timestamps))
            if timestamps[idx] <= timestamps[idx - 1]
        ]

        # Remove the problematic timestamps and corresponding data
        if drop_indices:
            # Log the warning with specific details
            logger.warning(
                "Found %d non-monotonic timestamp(s) that will be removed. "
                "Timestamps: %s. Indices: %s",
                len(drop_indices),
                timestamps[drop_indices],
                drop_indices,
            )

            # Optional: Log additional debugging information
            for idx in drop_indices:
                logger.debug(
                    "Dropping timestamp pair - Previous: %s, Current: %s at index %d",
                    timestamps[idx - 1],
                    timestamps[idx],
                    idx,
                )

            timestamps = np.delete(timestamps, drop_indices)
            positions = np.delete(positions, drop_indices, axis=0)
            rotations = Rotation.from_matrix(
                np.delete(homogeneous_matrix[:, :3, :3], drop_indices, axis=0)
            )

        self.pos_interp = interpolate.interp1d(
            timestamps, positions, kind="linear", axis=0
        )
        self.rot_interp = st.Slerp(timestamps, rotations)

    def __call__(self, ts):
        positions = self.pos_interp(ts)
        rotations = self.rot_interp(ts)
        rotation_matrix = rotations.as_matrix()
        homogeneous_matrix = np.array(
            [
                convert_homogeneous_matrix(R=rot, p=pos)
                for rot, pos in zip(rotation_matrix, positions)
            ]
        )
        return homogeneous_matrix


class ExoDataBuffer:
    def __init__(
        self,
        data_dir: str,
        target_dir: str,
        camera_latency: float = None,
        encoder_latency: float = None,
        tracking_latency: float = None,
        fsr_latencies: List[float] = None,
        enable_fsr: bool = False,
        num_fsr_sources: int = 3,  # Default to 3 FSR sources
        downsample_rate: int = 1,
    ) -> None:
        self.data_dir = data_dir
        self.target_dir = target_dir
        self.data_buffer = zarr.open(data_dir, mode="a")
        self.target_buffer = zarr.open(target_dir, mode="a")
        self.camera_latency = camera_latency
        self.encoder_latency = encoder_latency
        self.tracking_latency = tracking_latency

        # Initialize FSR latency handling for multiple sources
        if fsr_latencies is None:
            self.fsr_latencies = [None] * num_fsr_sources
        else:
            self.fsr_latencies = fsr_latencies
            # Ensure we have enough latencies for all FSR sources
            if len(self.fsr_latencies) < num_fsr_sources:
                self.fsr_latencies.extend(
                    [None] * (num_fsr_sources - len(self.fsr_latencies))
                )

        self.enable_fsr = enable_fsr
        self.num_fsr_sources = num_fsr_sources
        self.downsample_rate = downsample_rate

    def extract_frames_videos(self, video_path: str):
        frames = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break
        cap.release()
        return frames

    def interpolate_episode(self, episode_indices):
        for episode_index in episode_indices:
            episode_data = self.data_buffer[f"episode_{episode_index}"]
            target_episode_data = self.target_buffer.require_group(
                f"episode_{episode_index}"
            )
            self._interpolate_episode(episode_index, episode_data, target_episode_data)
            video_filename = os.path.join(
                self.data_dir, f"episode_{episode_index}", "camera_0.mp4"
            )
            target_filename = os.path.join(
                self.target_dir, f"episode_{episode_index}", "exo_camera_0.mp4"
            )
            os.makedirs(os.path.dirname(target_filename), exist_ok=True)
            if os.path.exists(video_filename):
                shutil.copy2(video_filename, target_filename)

    def _interpolate_episode(
        self,
        episode: int,
        episode_data: zarr.hierarchy.Group,
        target_episode_data: zarr.hierarchy.Group,
    ):
        camera_data = episode_data["camera_0"]
        if self.camera_latency:
            print("recalculating camera receive time")
            camera_capture_time = camera_data["receive_time"][:] - self.camera_latency
        else:
            camera_capture_time = camera_data["capture_time"][:]

        encoder_data = episode_data["numeric_0"]
        joint_angles = encoder_data["joint_angles"][:]
        if self.encoder_latency:
            encoder_angles_capture_time = (
                encoder_data["receive_time"][:] - self.encoder_latency
            )
        else:
            encoder_angles_capture_time = encoder_data["capture_time"][:]

        # Process multiple FSR data sources if enabled
        fsr_data_sources = []
        fsr_values_list = []
        fsr_capture_times = []

        if self.enable_fsr:
            for i in range(1, self.num_fsr_sources + 1):
                fsr_source_name = f"numeric_{i}"
                if fsr_source_name in episode_data:
                    fsr_data = episode_data[fsr_source_name]
                    fsr_values = fsr_data["fsr_values"][:]

                    # Apply latency correction if specified
                    latency_idx = i - 1  # Convert to 0-based index
                    if (
                        latency_idx < len(self.fsr_latencies)
                        and self.fsr_latencies[latency_idx] is not None
                    ):
                        fsr_capture_time = (
                            fsr_data["receive_time"][:]
                            - self.fsr_latencies[latency_idx]
                        )
                    else:
                        fsr_capture_time = fsr_data["capture_time"][:]

                    fsr_data_sources.append(fsr_data)
                    fsr_values_list.append(fsr_values)
                    fsr_capture_times.append(fsr_capture_time)
                else:
                    logger.warning(
                        f"FSR source {fsr_source_name} not found in episode {episode}"
                    )

        tracking_data = episode_data["camera_1"]
        if self.tracking_latency:
            tracking_capture_time = (
                tracking_data["receive_time"][:] - self.tracking_latency
            )
        else:
            tracking_capture_time = tracking_data["capture_time"][:]
        pose = tracking_data["pose"][:]

        # Find valid time range - include all FSR sources in the calculation
        time_mins = [
            camera_capture_time.min(),
            encoder_angles_capture_time.min(),
            tracking_capture_time.min(),
        ]
        time_maxs = [
            camera_capture_time.max(),
            encoder_angles_capture_time.max(),
            tracking_capture_time.max(),
        ]

        if self.enable_fsr and fsr_capture_times:
            # Add min/max times from all FSR sources
            for fsr_time in fsr_capture_times:
                time_mins.append(fsr_time.min())
                time_maxs.append(fsr_time.max())

        min_time = max(time_mins)
        max_time = min(time_maxs)

        # Filter camera timestamps to be within range
        # Create initial boolean mask for time validity
        valid_indices = (camera_capture_time >= min_time) & (
            camera_capture_time <= max_time
        )

        # Create a boolean array of the same length, all False
        downsample_mask = np.zeros_like(valid_indices, dtype=bool)

        # Set True for indices that should be kept after downsampling
        downsample_mask[:: self.downsample_rate] = True

        # Combine both conditions - must be within time range AND be a kept frame
        valid_indices = valid_indices & downsample_mask
        valid_camera_times = camera_capture_time[valid_indices]

        # Create interpolation function and interpolate only valid times
        f_encoder = interpolate.interp1d(
            encoder_angles_capture_time, joint_angles, kind="linear", axis=0
        )
        joint_angles_interp = f_encoder(valid_camera_times)
        joint_angles_interp = np.round(joint_angles_interp).astype(np.int32)
        f_pose = PoseInterpolator(tracking_capture_time, pose)
        pose_interp = f_pose(valid_camera_times)

        # Interpolate each FSR source
        fsr_values_interp_list = []
        if self.enable_fsr and fsr_values_list:
            for i, (fsr_values, fsr_time) in enumerate(
                zip(fsr_values_list, fsr_capture_times)
            ):
                f_fsr = interpolate.interp1d(
                    fsr_time, fsr_values, kind="linear", axis=0
                )
                fsr_values_interp = f_fsr(valid_camera_times)
                fsr_values_interp_list.append(fsr_values_interp)

        print(
            f"Original shape: {len(camera_capture_time)}, After filtering and downsampling: {len(valid_camera_times)}"
        )

        # Clean up previous interpolated data
        try:
            del encoder_data["joint_angles_interp"]
            del tracking_data["pose_interp"]
            del camera_data["valid_indices"]

            # Clean up previous FSR interpolated data
            if self.enable_fsr:
                for i, fsr_data in enumerate(fsr_data_sources):
                    if "fsr_values_interp" in fsr_data:
                        del fsr_data["fsr_values_interp"]
        except KeyError:
            pass

        # The interpolated joint_angles and pose data is already in valid camera times
        camera_data["valid_indices"] = valid_indices
        encoder_data["joint_angles_interp"] = joint_angles_interp
        tracking_data["pose_interp"] = pose_interp

        # Store interpolated FSR data
        if self.enable_fsr:
            for i, (fsr_data, fsr_values_interp) in enumerate(
                zip(fsr_data_sources, fsr_values_interp_list)
            ):
                fsr_data["fsr_values_interp"] = fsr_values_interp
                # Also store in target buffer
                target_episode_data[f"fsr_values_interp_{i + 1}"] = fsr_values_interp

        # Prepare replay data rgb and depth from tracking data
        first_frame_timestamp = valid_camera_times[0]
        closest_tracking_frame_index = np.argmin(
            np.abs(tracking_capture_time - first_frame_timestamp)
        )
        try:
            replay_rgb = extract_frames_videos(
                video_path=os.path.join(
                    self.data_dir, f"episode_{episode}", "camera_1.mp4"
                ),
            )[closest_tracking_frame_index]
            replay_depth = tracking_data["depth"][closest_tracking_frame_index]
            target_episode_data["replay_rgb"] = replay_rgb
            target_episode_data["replay_depth"] = replay_depth
            target_episode_data["replay_intrinsic"] = tracking_data["intrinsics"][
                closest_tracking_frame_index
            ]

        except KeyError:
            print("no replay data")

        # Prepare training data
        target_episode_data["valid_indices"] = valid_indices
        target_episode_data["pose_interp"] = pose_interp
        target_episode_data["joint_angles_interp"] = joint_angles_interp

        if (
            np.isclose(joint_angles, 0, atol=20).any()
            or np.isclose(joint_angles, 360, atol=20).any()
        ):
            print(f"episode {episode} Joint angles are close to 0 or 360 degrees")
            # get input
            input(
                "Warning: Joint angles are close to 0 or 360 degrees. Press Enter to continue..."
            )
            self.visualize_interpolation(episode=episode)

    def get_time_boundary(self, timestamp: np.array):
        time_lower_bound = np.min(timestamp)
        time_upper_bound = np.max(timestamp)
        return np.array([time_lower_bound, time_upper_bound])

    def visualize_interpolation(
        self,
        episode: int,
        save_path: Optional[str] = None,
    ):
        """Visualize raw and interpolated data for an episode.

        Args:
            episode: Episode number to visualize
            save_path: Optional path to save the plots
        """
        episode_data = self.data_buffer[f"episode_{episode}"]

        # Get raw data
        camera_data = episode_data["camera_0"]
        encoder_data = episode_data["numeric_0"]
        tracking_data = episode_data["camera_1"]
        valid_indices = camera_data["valid_indices"][:]

        # Get timestamps
        if self.camera_latency:
            camera_times = camera_data["receive_time"][:] - self.camera_latency
        else:
            camera_times = camera_data["capture_time"][:]

        if self.encoder_latency:
            encoder_times = encoder_data["receive_time"][:] - self.encoder_latency
        else:
            encoder_times = encoder_data["capture_time"][:]

        if self.tracking_latency:
            tracking_times = tracking_data["receive_time"][:] - self.tracking_latency
        else:
            tracking_times = tracking_data["capture_time"][:]

        # Get FSR times if FSR is enabled
        fsr_times_list = []
        if self.enable_fsr:
            for idx in range(1, self.num_fsr_sources + 1):
                fsr_source_name = f"numeric_{idx}"
                if fsr_source_name in episode_data:
                    fsr_data = episode_data[fsr_source_name]

                    # Get appropriate latency for this FSR source
                    latency_idx = idx - 1  # Convert to 0-based index
                    if (
                        latency_idx < len(self.fsr_latencies)
                        and self.fsr_latencies[latency_idx] is not None
                    ):
                        fsr_times = (
                            fsr_data["receive_time"][:]
                            - self.fsr_latencies[latency_idx]
                        )
                    else:
                        fsr_times = fsr_data["capture_time"][:]

                    fsr_times_list.append(fsr_times)

        # Find common x-axis limits
        all_times = [camera_times, encoder_times, tracking_times] + fsr_times_list
        x_min = min(np.min(times) for times in all_times)
        x_max = max(np.max(times) for times in all_times)

        # Determine number of subplots based on FSR enabled status - additional plot for each FSR source
        n_plots = 3 + (self.num_fsr_sources if self.enable_fsr else 0)

        # Create figure with shared x-axis
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 4 * n_plots), sharex=True)

        # Get raw and interpolated data
        joint_angles = encoder_data["joint_angles"][:]
        joint_angles_interp = encoder_data["joint_angles_interp"][:]

        raw_pose = tracking_data["pose"][:]
        pose_interp = tracking_data["pose_interp"][:]

        # Plot Joint_angles
        for i in range(11, 12):  # Plot all 6 Joint_angles channels
            axes[0].plot(
                encoder_times, joint_angles[:, i], ".", label=f"Joint_angles {i}"
            )
            axes[0].plot(
                camera_times[valid_indices],
                joint_angles_interp[:, i],
                "-",
                label=f"Interp joint angles {i}",
            )
        axes[0].set_title(f"Episode: {episode}  Encoder joint angles")
        axes[0].legend()

        # Plot translations
        raw_translations = raw_pose[:, :3, 3]
        interp_translations = pose_interp[:, :3, 3]

        for i, coord in enumerate(["X", "Y", "Z"]):
            axes[1].plot(
                tracking_times, raw_translations[:, i], ".", label=f"Raw {coord}"
            )
            axes[1].plot(
                camera_times[valid_indices],
                interp_translations[:, i],
                "-",
                label=f"Interp {coord}",
            )
        axes[1].set_title("Translations")
        axes[1].legend()

        # Plot rotations as euler angles
        raw_rotations = Rotation.from_matrix(raw_pose[:, :3, :3]).as_euler("xyz")
        interp_rotations = Rotation.from_matrix(pose_interp[:, :3, :3]).as_euler("xyz")

        for i, angle in enumerate(["Roll", "Pitch", "Yaw"]):
            axes[2].plot(
                tracking_times,
                np.rad2deg(raw_rotations[:, i]),
                ".",
                label=f"Raw {angle}",
            )
            axes[2].plot(
                camera_times[valid_indices],
                np.rad2deg(interp_rotations[:, i]),
                "-",
                label=f"Interp {angle}",
            )
        axes[2].set_title("Rotation (Euler angles)")
        axes[2].legend()

        # Plot FSR values for each source if enabled
        if self.enable_fsr:
            for idx in range(1, self.num_fsr_sources + 1):
                fsr_source_name = f"numeric_{idx}"
                if fsr_source_name in episode_data:
                    fsr_data = episode_data[fsr_source_name]

                    # Get appropriate latency for this FSR source
                    latency_idx = idx - 1  # Convert to 0-based index
                    if (
                        latency_idx < len(self.fsr_latencies)
                        and self.fsr_latencies[latency_idx] is not None
                    ):
                        fsr_times = (
                            fsr_data["receive_time"][:]
                            - self.fsr_latencies[latency_idx]
                        )
                    else:
                        fsr_times = fsr_data["capture_time"][:]

                    raw_fsr = fsr_data["fsr_values"][:]
                    fsr_interp = fsr_data["fsr_values_interp"][:]

                    # Plot in the appropriate subplot (3 + source_idx - 1)
                    plot_idx = 3 + idx - 1
                    for i in range(raw_fsr.shape[1]):
                        axes[plot_idx].plot(
                            fsr_times, raw_fsr[:, i], ".", label=f"Raw FSR {i}"
                        )
                        axes[plot_idx].plot(
                            camera_times[valid_indices],
                            fsr_interp[:, i],
                            "-",
                            label=f"Interp FSR {i}",
                        )
                    axes[plot_idx].set_title(f"FSR Values - Source {idx}")
                    axes[plot_idx].legend()

        # Set common x-axis limits for all subplots
        for ax in axes:
            ax.set_xlim(x_min, x_max)

        # Add a common x-axis label at the bottom
        fig.text(0.5, 0.04, "Time", ha="center", va="center")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()
