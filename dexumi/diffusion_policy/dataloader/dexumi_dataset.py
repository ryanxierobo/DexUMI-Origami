import numpy as np
from dexumi.common.utility.matrix import (
    homogeneous_matrix_to_6dof,
    relative_transformation,
    vec6dof_to_homogeneous_matrix,
)
from tqdm import tqdm

from .diffusion_bc_dataset import (
    DiffusionBCDataset,
    get_data_stats,
    normalize_data,
    process_image,
    sample_sequence,
)
from .replay_buffer import DexUMIReplayBuffer


class DexUMIDataset(DiffusionBCDataset):
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
        optional_transforms=None,
        replay_buffer_cls=DexUMIReplayBuffer,
        relative_hand_action=False,
        **replay_buffer_kwargs,
    ):
        self.relative_hand_action = relative_hand_action

        if self.relative_hand_action:
            if "hand_action" not in unnormal_list:
                unnormal_list.append("hand_action")
                print("Adding hand_action to unnormal_list")
            assert "hand_action" in unnormal_list
        else:
            if "hand_action" in unnormal_list:
                unnormal_list.remove("hand_action")
                print("Removing hand_action from unnormal_list")
            assert "hand_action" not in unnormal_list
        assert "pose" in unnormal_list
        super().__init__(
            data_dirs,
            max_episode,
            load_camera_ids,
            camera_resize_shape,
            pred_horizon,
            obs_horizon,
            action_horizon,
            unnormal_list,
            seed,
            replay_buffer_cls,
            **replay_buffer_kwargs,
        )
        self.optional_transforms = optional_transforms
        self.get_relative_action_normalization_stats()

    def get_relative_action_normalization_stats(self):
        print("Computing relative normalization stats")
        all_relative_vec6dof = []
        all_relative_hand_action = []
        for idx in tqdm(range(len(self))):
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
            # relative pose
            xyzNrotvec = nsample["pose"]
            Ts = [
                vec6dof_to_homogeneous_matrix(
                    translation=xyzNrotvec[i, :3], rotation_vector=xyzNrotvec[i, 3:]
                )
                for i in range(self.pred_horizon)
            ]
            relative_transformation_Ts = [relative_transformation(Ts[0], T) for T in Ts]
            relative_vec6dof = np.array(
                [homogeneous_matrix_to_6dof(T) for T in relative_transformation_Ts],
                dtype=np.float32,
            )
            all_relative_vec6dof.append(relative_vec6dof)
            # relative hand action
            if self.relative_hand_action:
                # TODO: check if this is correct
                hand_action = nsample["hand_action"]
                relative_hand_action = [h - hand_action[0] for h in hand_action]
                relative_hand_action = np.array(relative_hand_action, dtype=np.float32)
                all_relative_hand_action.append(relative_hand_action)
        all_relative_vec6dof = np.array(all_relative_vec6dof)
        print("all_relative pose shape", all_relative_vec6dof.shape)
        self.stats["relative_pose"] = get_data_stats(all_relative_vec6dof)
        print("relative pose stats", self.stats["relative_pose"])
        if self.relative_hand_action:
            all_relative_hand_action = np.array(all_relative_hand_action)
            print("all_relative hand action shape", all_relative_hand_action.shape)
            self.stats["relative_hand_action"] = get_data_stats(
                all_relative_hand_action
            )
            print("relative hand action stats", self.stats["relative_hand_action"])

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
            nsample[f"camera_{camera_id}"] = process_image(
                nsample[f"camera_{camera_id}"][: self.obs_horizon, :],
                self.optional_transforms,
                resize_shape=self.buffer.camera_resize_shape,
            )
        # get the relative transformation between the first and current frame
        xyzNrotvec, hand_action = nsample["pose"], nsample["hand_action"]
        Ts = [
            vec6dof_to_homogeneous_matrix(
                translation=xyzNrotvec[i, :3], rotation_vector=xyzNrotvec[i, 3:]
            )
            for i in range(self.pred_horizon)
        ]
        relative_transformation_Ts = [relative_transformation(Ts[0], T) for T in Ts]
        relative_vec6dof = np.array(
            [homogeneous_matrix_to_6dof(T) for T in relative_transformation_Ts],
            dtype=np.float32,
        )
        relative_vec6dof = normalize_data(relative_vec6dof, self.stats["relative_pose"])
        if self.relative_hand_action:
            relative_hand_action = [h - hand_action[0] for h in hand_action]
            relative_hand_action = np.array(relative_hand_action, dtype=np.float32)
            relative_hand_action = normalize_data(
                relative_hand_action, self.stats["relative_hand_action"]
            )
            nsample["action"] = np.concatenate(
                [relative_vec6dof, relative_hand_action], axis=1
            )
        else:
            # if not, simply concatenate the relative pose and absolute hand_action
            nsample["action"] = np.concatenate([relative_vec6dof, hand_action], axis=1)
        # discard unused observations
        if not self.buffer.skip_proprioception:
            nsample["proprioception"] = nsample["proprioception"][: self.obs_horizon, :]

        if self.buffer.enable_fsr:
            nsample["fsr"] = nsample["fsr"][: self.obs_horizon, :]

        return nsample
