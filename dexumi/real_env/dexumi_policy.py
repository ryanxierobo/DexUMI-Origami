import os

import cv2
import numpy as np
import torch
from dexumi.common.utility.file import read_pickle
from dexumi.common.utility.model import load_config, load_diffusion_model
from dexumi.constants import INPAINT_RESIZE_RATIO
from dexumi.diffusion_policy.dataloader.diffusion_bc_dataset import (
    normalize_data,
    process_image,
    unnormalize_data,
)
from dexumi.real_env.common.policy import PolicyClient, PolicyServer
from matplotlib import pyplot as plt


class DexUMIPolicySever(PolicyServer):
    def __init__(
        self,
        obs_config,
        model_path,
        ckpt,
        req_address: str = "ipc:///tmp/policy_req",
    ):
        super().__init__(
            obs_config=obs_config,
            req_address=req_address,
        )
        self.obs_config = obs_config
        model_cfg = load_config(model_path)
        model, noise_scheduler = load_diffusion_model(model_path, ckpt)
        stats = read_pickle(os.path.join(model_path, "stats.pickle"))
        self.pred_horizon = model_cfg.dataset.pred_horizon
        self.action_dim = model_cfg.action_dim
        self.obs_horizon = model_cfg.dataset.obs_horizon

        self.model = model.eval()
        self.noise_scheduler = noise_scheduler
        self.num_inference_steps = model_cfg.num_inference_steps
        self.stats = stats
        if model_cfg.dataset.relative_hand_action:
            print("Using relative hand action")
            print("hand_action stats", stats["relative_hand_action"])
            print(
                stats["relative_hand_action"]["max"]
                - stats["relative_hand_action"]["min"]
                > 5e-2
            )
            self.stats["action"] = {
                "min": np.concatenate(
                    [
                        stats["relative_pose"]["min"],
                        stats["relative_hand_action"]["min"],
                    ]
                ),
                "max": np.concatenate(
                    [
                        stats["relative_pose"]["max"],
                        stats["relative_hand_action"]["max"],
                    ]
                ),
            }
        else:
            print("Using absolute hand action")
            print("hand_action stats", stats["hand_action"])
            print(stats["hand_action"]["max"] - stats["hand_action"]["min"] > 5e-2)
            self.stats["action"] = {
                "min": np.concatenate(
                    [stats["relative_pose"]["min"], stats["hand_action"]["min"]]
                ),
                "max": np.concatenate(
                    [stats["relative_pose"]["max"], stats["hand_action"]["max"]]
                ),
            }
        print(self.stats["action"])
        self.model_cfg = model_cfg

    def _preprocess_policy_obs(self, policy_obs: dict):
        proprioception = policy_obs.get("proprioception", None)
        visual_obs = policy_obs.get("visual_obs")
        fsr = policy_obs.get("fsr", None)
        if proprioception is not None:
            proprioception = normalize_data(
                proprioception.reshape(1, -1), self.stats["proprioception"]
            )  # (1,N)
            proprioception = (
                torch.from_numpy(proprioception).unsqueeze(0).cuda()
            )  # (B,1,7)
        if fsr is not None:
            fsr = normalize_data(fsr.reshape(1, -1), self.stats["fsr"])
            fsr = torch.from_numpy(fsr).unsqueeze(0).cuda()  # (B,1,2)

        _, H, W, _ = visual_obs.shape
        visual_obs = np.array(
            [
                cv2.cvtColor(
                    cv2.resize(
                        obs,
                        (
                            int(W * INPAINT_RESIZE_RATIO),
                            int(H * INPAINT_RESIZE_RATIO),
                        ),
                    ),
                    cv2.COLOR_BGR2RGB,
                )
                for obs in visual_obs
            ]
        )

        visual_obs = process_image(
            visual_obs,
            optional_transforms=["Resize", "CenterCrop"],
        )
        visual_obs = visual_obs.unsqueeze(0).cuda()
        return {"proprioception": proprioception, "visual_obs": visual_obs, "fsr": fsr}

    def _inference_action(self, policy_obs):
        proprioception = policy_obs.get("proprioception")
        visual_obs = policy_obs.get("visual_obs")
        fsr = policy_obs.get("fsr")
        B = visual_obs.shape[0]
        assert B == 1
        trajectory = torch.randn(B, self.pred_horizon, self.action_dim).cuda()
        trajectory = self.model.inference(
            proprioception=proprioception,
            fsr=fsr,
            visual_obs=visual_obs,
            trajectory=trajectory,
            noise_scheduler=self.noise_scheduler,
            num_inference_steps=self.num_inference_steps,
        )
        trajectory = trajectory.detach().to("cpu").numpy()
        naction = trajectory[0]
        action_pred = unnormalize_data(naction, stats=self.stats["action"])
        start = self.obs_horizon - 1
        end = start + self.pred_horizon
        action = action_pred[start:end, :]
        return action


# class DexUMIPolicyClient(PolicyClient):
#     def __init__(
#         self,
#         pub_address: str = "ipc:///tmp/policy_pub",
#         req_address: str = "ipc:///tmp/policy_req",
#         req_frequency: int = 1000,
#         topic: str = "policy",
#         verbose: bool = True,
#     ):
#         super().__init__(
#             pub_address=pub_address,
#             req_address=req_address,
#             topic=topic,
#             req_frequency=req_frequency,
#             verbose=verbose,
#         )
#         self.model_cfg = self.get_attr("model_cfg")
