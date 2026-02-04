import random

import numpy as np
import torch
from torchvision.transforms import v2

from .replay_buffer import ReplayBuffer

normalize_threshold = 5e-2


def process_image(image, optional_transforms=[], resize_shape=(240, 240)):
    if isinstance(image, np.ndarray):
        if image.ndim not in [3, 4]:
            raise ValueError("Image must be 3D (H,W,C) or 4D (N,H,W,C)")
        image = torch.from_numpy(image)
        if image.ndim == 3:
            image = image.permute(2, 0, 1)  # (C,H,W)
        else:
            image = image.permute(0, 3, 1, 2)  # (N,C,H,W)

    transform_list = [
        v2.ToDtype(torch.float32, scale=True),  # convert to float32 and scale to [0,1]
    ]

    for transform_name in optional_transforms:
        if transform_name == "Resize":
            transform_list.append(v2.Resize(resize_shape))
        if transform_name == "RandomCrop":
            transform_list.append(v2.RandomCrop((224, 224)))
        elif transform_name == "CenterCrop":
            transform_list.append(v2.CenterCrop((224, 224)))

        if transform_name == "GaussianBlur":
            transform_list.append(
                v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.2)
            )
        if transform_name == "ColorJitter":
            transform_list.append(
                v2.RandomApply(
                    [
                        v2.ColorJitter(
                            brightness=0.64, contrast=0.32, saturation=0.32, hue=0.08
                        )
                    ],
                    p=0.8,
                )
            )
        if transform_name == "RandomGrayscale":
            transform_list.append(v2.RandomGrayscale(p=0.2))

    transform_list.append(
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )

    transforms = v2.Compose(transform_list)
    return transforms(image)


def create_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
            )
    indices = np.array(indices)
    return indices


def sample_sequence(
    train_data,
    sequence_length,
    buffer_start_idx,
    buffer_end_idx,
    sample_start_idx,
    sample_end_idx,
):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype
            )
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


# normalize data
def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}
    return stats


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = data.copy()
    for i in range(ndata.shape[1]):
        if stats["max"][i] - stats["min"][i] > normalize_threshold:
            ndata[:, i] = (data[:, i] - stats["min"][i]) / (
                stats["max"][i] - stats["min"][i]
            )
            # normalize to [-1, 1]
            ndata[:, i] = ndata[:, i] * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    data = ndata.copy()
    for i in range(ndata.shape[1]):
        if stats["max"][i] - stats["min"][i] > normalize_threshold:
            ndata[:, i] = (ndata[:, i] + 1) / 2
            data[:, i] = (
                ndata[:, i] * (stats["max"][i] - stats["min"][i]) + stats["min"][i]
            )
    return data


class DiffusionBCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dirs,
        max_episode=None,
        load_camera_ids=[0],
        camera_resize_shape=None,
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=16,
        unnormal_list=[],
        seed=0,
        replay_buffer_cls=ReplayBuffer,
        **replay_buffer_kwargs,
    ):
        self.buffer = replay_buffer_cls(
            data_dirs,
            load_camera_ids,
            camera_resize_shape,
            max_episode=max_episode,
            **replay_buffer_kwargs,
        )
        self.seed = seed
        self.set_seed(self.seed)
        self.unnormal_list = unnormal_list
        episode_ends = self.buffer.eps_end
        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            # add padding such that each timestep in the dataset are seen
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        # normalized_train_data = dict()
        for key, data in self.buffer.memory_buffer.items():
            if key in self.unnormal_list:
                pass
            else:
                stats[key] = get_data_stats(data)

            if key in self.unnormal_list:
                pass
            else:
                self.buffer.memory_buffer[key] = normalize_data(data, stats[key])

        self.indices = indices
        self.stats = stats
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def transform_images(self, images_arr):
        images_arr = images_arr.astype(np.float32)
        images_tensor = np.transpose(images_arr, (0, 3, 1, 2)) / 255.0  # (T,dim,h,w)
        return images_tensor

    def __len__(self):
        # all possible segments of the dataset
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
        ) = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.buffer.memory_buffer,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )
        for camera_id in self.buffer.load_camera_ids:
            nsample[f"camera_{camera_id}"] = self.transform_images(
                nsample[f"camera_{camera_id}"][: self.obs_horizon, :]
            )

        # discard unused observations
        nsample["proprioception"] = nsample["proprioception"][: self.obs_horizon, :]

        return nsample
