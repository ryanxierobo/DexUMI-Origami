import logging
import os
import subprocess

import hydra
import omegaconf
import zarr
from dexumi.common.utility.logging import setup_logging
from dexumi.common.utility.parallel import assign_task_bounds_to_gpus
from dexumi.common.utility.zarr import get_episode_num, get_episode_number


@hydra.main(
    version_base=None,
    config_path="../../config/render",
    config_name="render_all_dataset",
)
def main(cfg: omegaconf.DictConfig):
    print(cfg)
    available_gpu = cfg.avaliable_gpu
    num_gpu = len(available_gpu)
    print(cfg.data_buffer_path)
    data_buffer = zarr.open(cfg.data_buffer_path, mode="a")
    if cfg.start_episode is not None and cfg.end_episode is not None:
        num_episode = cfg.end_episode - cfg.start_episode
        print("Using start and end episode from config")
    else:
        num_episode = get_episode_num(data_buffer)
    task_bounds = assign_task_bounds_to_gpus(
        num_episode, num_gpu, start_id=cfg.start_episode if cfg.start_episode else 0
    )

    processes = []
    for i, (start, end) in enumerate(task_bounds):
        gpu_id = available_gpu[i]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cmd = [
            "python",
            "render_dataset.py",
            "--start",
            str(start),
            "--end",
            str(end),
            "--gpu_id",
            str(gpu_id),
            "--data_dir",
            str(cfg.data_dir),
            "--reference_dir",
            str(cfg.reference_dir),
            "--sam2_checkpoint_path",
            str(cfg.sam2_checkpoint_path),
            "--save_path",
            str(cfg.save_path),
            "--resize_ratio",
            str(cfg.resize_ratio),
        ]

        process = subprocess.Popen(cmd, env=env)
        processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        process.wait()


if __name__ == "__main__":
    main()
