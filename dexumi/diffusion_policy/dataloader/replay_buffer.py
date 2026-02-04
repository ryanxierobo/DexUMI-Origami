import concurrent.futures
import os.path as osp
from collections import defaultdict

import cv2
import numpy as np
import zarr
from dexumi.common.imagecodecs_numcodecs import register_codecs
from tqdm import tqdm

register_codecs()


class ReplayBuffer:
    def __init__(
        self,
        data_path,
        load_camera_ids=[],
        camera_resize_shape=[],
        max_episode=None,
        max_workers=4,
        bgr2rgb=False,  # Add new parameter
    ) -> None:
        self.data_path = data_path
        self.load_camera_ids = load_camera_ids
        self.camera_resize_shape = camera_resize_shape
        self.max_workers = max_workers
        self.max_episode = max_episode
        self.bgr2rgb = bgr2rgb  # Store parameter
        self.initiate_memory_buffer()
        self.load_data_to_memory()

    def initiate_memory_buffer(self):
        self.memory_buffer = defaultdict(list)

    def load_data_to_memory(self):
        load_episode_num = 0
        for path in self.data_path:
            root = zarr.open(path, mode="r")
            episodes = list(root.group_keys())
            for episode in episodes:
                self.memory_buffer["action"].append(
                    self.load_low_dim_data(root, osp.join(episode, "action"))
                )
                self.memory_buffer["proprioception"].append(
                    self.load_low_dim_data(root, osp.join(episode, "proprioception"))
                )
                for camera_ids in self.load_camera_ids:
                    cam_name = f"camera_{camera_ids}"
                    self.memory_buffer[cam_name].append(
                        self.load_visual_data(root, osp.join(episode, cam_name, "rgb"))
                    )
                load_episode_num += 1
                if (
                    self.max_episode is not None
                    and load_episode_num >= self.max_episode
                ):
                    break

        self.eps_end = np.cumsum([len(x) for x in self.memory_buffer["action"]])
        for k, v in self.memory_buffer.items():
            self.memory_buffer[k] = np.concatenate(v)

    def load_low_dim_data(self, root, low_dim_path):
        return root[low_dim_path][:].astype(np.float32)

    def load_visual_data(self, root, visual_path, dim=3):
        visual_shape = root[visual_path].shape
        np_arr_shape = (
            (visual_shape[0], *self.camera_resize_shape, dim)
            if self.camera_resize_shape
            else visual_shape
        )
        np_arr = np.zeros(np_arr_shape, dtype=np.uint8)

        def load_img(zarr_arr, visual_path, index, np_arr):
            try:
                img = zarr_arr[visual_path][index]
                if self.camera_resize_shape:
                    img = cv2.resize(img, self.camera_resize_shape)
                if self.bgr2rgb:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                np_arr[index] = img
                return True
            except Exception as e:
                print(e)
                return False

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = set()
            for i in range(visual_shape[0]):
                futures.add(executor.submit(load_img, root, visual_path, i, np_arr))

            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result():
                    raise RuntimeError("Failed to load image!")

        return np_arr

    def __repr__(self) -> str:
        rep = ""
        for k, v in self.memory_buffer.items():
            rep += f"{k}, {v.shape}\n"
        rep += f"eps_end, {self.eps_end}\n"
        return rep

    def __getitem__(self, key):
        return self.memory_buffer[key]

    def remove_key(self, key):
        del self.memory_buffer[key]


class DexUMIReplayBuffer(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        # Extract optional parameters
        self.skip_proprioception = kwargs.pop("skip_proprioception", False)

        # Handle FSR options
        self.enable_fsr = False
        if kwargs.get("enable_fsr", False):
            self.enable_fsr = kwargs.pop("enable_fsr")
            self.fsr_binary_cutoff = kwargs.pop("fsr_binary_cutoff")
            self.fsr_binary_cutoff = np.array(self.fsr_binary_cutoff)
            print("===============")
            print("enable_fsr with binary cutoff", self.fsr_binary_cutoff)
            print("===============")
        else:
            # Clean up kwargs if keys exist
            kwargs.pop("enable_fsr", None)
            kwargs.pop("fsr_binary_cutoff", None)

        # Initialize parent class
        super().__init__(*args, **kwargs)

    def load_data_to_memory(self):
        # Use a generator to avoid loading all episodes at once
        episode_lengths = []
        total_frames = 0

        # First pass: count total frames and get episode lengths
        for path in self.data_path:
            root = zarr.open(path, mode="r")
            episodes = list(root.group_keys())

            for episode_idx, episode in enumerate(
                tqdm(episodes, desc="Scanning episodes")
            ):
                # Get episode length from pose data
                pose_path = osp.join(episode, "pose")
                episode_length = len(root[pose_path])
                episode_lengths.append(episode_length)
                total_frames += episode_length

                if self.max_episode is not None and episode_idx + 1 >= self.max_episode:
                    break

        # Pre-allocate arrays with the exact size needed
        self._preallocate_arrays(total_frames)

        # Second pass: load the data
        load_episode_num = 0
        current_idx = 0

        for path in self.data_path:
            root = zarr.open(path, mode="r")
            episodes = list(root.group_keys())

            for episode in tqdm(episodes, desc="Loading episodes"):
                episode_length = episode_lengths[load_episode_num]
                end_idx = current_idx + episode_length

                # Load hand action
                self.memory_buffer["hand_action"][current_idx:end_idx] = (
                    self.load_low_dim_data(root, osp.join(episode, "hand_action"))
                )

                # Load pose
                self.memory_buffer["pose"][current_idx:end_idx] = (
                    self.load_low_dim_data(root, osp.join(episode, "pose"))
                )

                # Load proprioception if needed
                if not self.skip_proprioception:
                    self.memory_buffer["proprioception"][current_idx:end_idx] = (
                        self.load_low_dim_data(
                            root, osp.join(episode, "proprioception")
                        )
                    )

                # Load FSR if enabled
                if self.enable_fsr:
                    fsr_data = self.load_low_dim_data(root, osp.join(episode, "fsr"))
                    # Binarize FSR data using the cutoff values
                    fsr_data = np.where(fsr_data >= self.fsr_binary_cutoff, 1.0, 0.0)
                    fsr_data = fsr_data.astype(np.float32)
                    self.memory_buffer["fsr"][current_idx:end_idx] = fsr_data

                # Load camera data
                for camera_id in self.load_camera_ids:
                    cam_name = f"camera_{camera_id}"
                    self.memory_buffer[cam_name][current_idx:end_idx] = (
                        self.load_visual_data(root, osp.join(episode, cam_name, "rgb"))
                    )

                # Update episode end indices
                if load_episode_num == 0:
                    self.eps_end = [episode_length]
                else:
                    self.eps_end.append(self.eps_end[-1] + episode_length)

                current_idx = end_idx
                load_episode_num += 1

                if (
                    self.max_episode is not None
                    and load_episode_num >= self.max_episode
                ):
                    break

            if self.max_episode is not None and load_episode_num >= self.max_episode:
                break

        # Convert eps_end to numpy array for faster indexing
        self.eps_end = np.array(self.eps_end)

        print(f"Loaded {load_episode_num} episodes with {total_frames} frames")

    def _preallocate_arrays(self, total_frames):
        """Pre-allocate arrays with the exact size needed to avoid memory fragmentation"""
        # Get sample shapes by loading first frame of first episode
        root = zarr.open(self.data_path[0], mode="r")
        episode = list(root.group_keys())[0]

        # Pre-allocate arrays for actions and pose
        hand_action_shape = root[osp.join(episode, "hand_action")].shape[1:]
        pose_shape = root[osp.join(episode, "pose")].shape[1:]

        self.memory_buffer["hand_action"] = np.zeros(
            (total_frames, *hand_action_shape), dtype=np.float32
        )
        self.memory_buffer["pose"] = np.zeros(
            (total_frames, *pose_shape), dtype=np.float32
        )

        # Pre-allocate array for proprioception if needed
        if not self.skip_proprioception:
            proprio_shape = root[osp.join(episode, "proprioception")].shape[1:]
            self.memory_buffer["proprioception"] = np.zeros(
                (total_frames, *proprio_shape), dtype=np.float32
            )

        # Pre-allocate array for FSR if enabled
        if self.enable_fsr:
            fsr_shape = root[osp.join(episode, "fsr")].shape[1:]
            self.memory_buffer["fsr"] = np.zeros(
                (total_frames, *fsr_shape), dtype=np.float32
            )

        # Pre-allocate arrays for camera data
        for camera_id in self.load_camera_ids:
            cam_name = f"camera_{camera_id}"
            rgb_path = osp.join(episode, cam_name, "rgb")

            # Get the shape of a single image
            img_shape = root[rgb_path].shape[1:]

            # Adjust shape if resize is required
            if self.camera_resize_shape:
                img_shape = (*self.camera_resize_shape, img_shape[2])

            # Pre-allocate camera array
            self.memory_buffer[cam_name] = np.zeros(
                (total_frames, *img_shape), dtype=np.uint8
            )

        # Initialize eps_end as a list
        self.eps_end = []
