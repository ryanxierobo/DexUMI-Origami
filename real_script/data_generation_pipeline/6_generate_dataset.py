import os
import traceback

import click
import numpy as np
import zarr
from einops import rearrange
from tqdm import tqdm

from dexumi.common.imagecodecs_numcodecs import JpegXl, register_codecs
from dexumi.common.utility.matrix import homogeneous_matrix_to_6dof
from dexumi.common.utility.parallel import calculate_processes_from_cpu_percentage
from dexumi.common.utility.video import (
    extract_frames_videos,
    load_images_from_folder,
)
from dexumi.common.utility.zarr import get_episode_num, parallel_saving
from dexumi.constants import (
    INPAINT_RESIZE_RATIO,
    INSPIRE_HAND_MOTOR_SCALE_FACTOR,
    XHAND_HAND_MOTOR_SCALE_FACTOR,
)

register_codecs()

# Get base exoskeleton directory path
exo_base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def compute_total_force_per_finger(all_fsr_observations):
    """
    Compute the total force for each finger.

    Parameters:
    all_fsr_observations (numpy.ndarray): Array of shape (m, n, 3) where:
        - m is the number of observations
        - n is the number of fingers
        - 3 is the xyz force components

    Returns:
    numpy.ndarray: Array of shape (m, n) with total force magnitude for each finger
    """
    # Calculate the Euclidean norm (magnitude) of the 3D force vector for each finger
    # This computes sqrt(x² + y² + z²) for each finger at each observation
    total_force = np.linalg.norm(all_fsr_observations, axis=2)

    return total_force


@click.command()
@click.option(
    "-d",
    "--data_dir",
    required=True,
    default="~/Dev/exoskeleton/data_local/exoskeleton_replay",
    help="Directory of JPEG frames.",
)
@click.option(
    "-t",
    "--target_dir",
    type=click.Path(),
    default="~/Dev/exoskeleton/data_local/dexumi_data",
    help="Base directory for dexumi data",
)
@click.option(
    "--hand-type",
    type=click.Choice(["inspire", "xhand"]),
    default="xhand",
    help="Type of hand to use",
)
@click.option(
    "--force-process",
    type=click.Choice(["flatten", "total"]),
    default="xhand",
    help="Type of hand to use",
)
@click.option(
    "-p",
    "--percentage-of-cores",
    default=70,
    help="Percentage of CPU cores to use",
)
@click.option(
    "--downsample-rate",
    type=int,
    default=1,
    help="Post process Downsample rate for data",
)
@click.option(
    "--start-episode",
    type=int,
    default=None,
    help="Starting episode number (inclusive)",
)
@click.option(
    "--end-episode",
    type=int,
    default=None,
    help="Ending episode number (exclusive)",
)
@click.option(
    "--generate-maskout-baseline-dataset",
    is_flag=True,
    default=False,
    help="Whether to generate baseline dataset",
)
@click.option(
    "--generate-exo-baseline-dataset",
    is_flag=True,
    default=False,
    help="Whether to generate baseline dataset",
)
@click.option(
    "--force-adjust",
    is_flag=True,
    default=False,
    help="Whether to subtract the fsr by the inital frame",
)
def main(
    data_dir,
    target_dir,
    hand_type,
    force_process,
    force_adjust,
    percentage_of_cores,
    downsample_rate,
    start_episode,
    end_episode,
    generate_maskout_baseline_dataset,
    generate_exo_baseline_dataset,
):
    if hand_type == "inspire":
        hand_motor_scale_factor = INSPIRE_HAND_MOTOR_SCALE_FACTOR
    elif hand_type == "xhand":
        hand_motor_scale_factor = XHAND_HAND_MOTOR_SCALE_FACTOR
    else:
        raise ValueError("Invalid hand type")
    assert data_dir != target_dir
    if not target_dir.endswith("dataset"):
        raise ValueError("target_dir must end with 'dataset'")
    if (
        generate_maskout_baseline_dataset or generate_exo_baseline_dataset
    ) and not target_dir.endswith("baseline_dataset"):
        raise ValueError(
            "target_dir must end with 'baseline_dataset' when generating baseline dataset"
        )
    max_workers = calculate_processes_from_cpu_percentage(percentage_of_cores)
    data_dir = os.path.expanduser(data_dir)
    target_dir = os.path.expanduser(target_dir)
    source_buffer = zarr.open(data_dir, mode="a")
    target_buffer = zarr.open(target_dir, mode="a")
    n_episode = get_episode_num(source_buffer)

    # Determine episode range
    start_idx = start_episode if start_episode is not None else 0
    end_idx = end_episode if end_episode is not None else n_episode

    valid_episode = 0
    failed_episodes = []
    for i in tqdm(range(start_idx, end_idx)):
        try:
            # action: tracking pose (6dim) + joint angles (6 dim for inspire and 12 dim for xhand)
            pose = source_buffer[f"episode_{i}/pose_interp"][:][::downsample_rate]
            xyzNrotvec = np.array([homogeneous_matrix_to_6dof(p) for p in pose])
            hand_motor_value = source_buffer[f"episode_{i}/hand_motor_value"][:][
                ::downsample_rate
            ]
            hand_motor_value = hand_motor_value.astype(np.float32)
            hand_motor_value = hand_motor_value * hand_motor_scale_factor
            assert xyzNrotvec.shape[0] == hand_motor_value.shape[0]
            # observation: both visual and fsr, and proprioception
            visual_observation = extract_frames_videos(
                os.path.join(data_dir, f"episode_{i}/combined.mp4"),
                BGR2RGB=True,
            )[::downsample_rate]
            visual_observation = np.array(visual_observation, dtype=np.uint8)[:-1]
            assert visual_observation.shape[0] == xyzNrotvec[1:].shape[0]
            # after validation, save to target buffer
            episode_data = target_buffer.require_group(f"episode_{valid_episode}")
            # save action
            episode_data["hand_action"] = hand_motor_value[1:]
            episode_data["pose"] = xyzNrotvec[1:]
            # save proprioception
            episode_data["proprioception"] = hand_motor_value[:-1]
            # save video as zarr
            episode_camera_data = episode_data.require_group("camera_0")
            this_compressor = JpegXl(level=80, numthreads=1)
            if not (generate_exo_baseline_dataset or generate_maskout_baseline_dataset):
                n, h, w, c = visual_observation.shape
                parallel_saving(
                    group=episode_camera_data,
                    array_name="rgb",
                    shape=(n, h, w, c),
                    chunks=(1, h, w, c),
                    dtype=np.uint8,
                    overwrite=True,
                    arr_to_save=visual_observation,
                    max_workers=max_workers,
                    compressor=this_compressor,
                )
            elif generate_exo_baseline_dataset:
                episode_exo_img = load_images_from_folder(
                    os.path.join(data_dir, f"episode_{i}/exo_img"),
                    BGR2RGB=True,
                    resize_ratio=INPAINT_RESIZE_RATIO,
                )[::downsample_rate]
                episode_exo_img = np.array(episode_exo_img, dtype=np.uint8)[:-1]
                assert episode_exo_img.shape[0] == xyzNrotvec[1:].shape[0]
                n, h, w, c = episode_exo_img.shape
                print(episode_exo_img.shape)
                parallel_saving(
                    group=episode_camera_data,
                    array_name="rgb",
                    shape=(n, h, w, c),
                    chunks=(1, h, w, c),
                    dtype=np.uint8,
                    overwrite=True,
                    arr_to_save=episode_exo_img,
                    max_workers=max_workers,
                    compressor=this_compressor,
                )
            elif generate_maskout_baseline_dataset:
                episode_mask_img = extract_frames_videos(
                    os.path.join(data_dir, f"episode_{i}/maskout_baseline.mp4"),
                    BGR2RGB=True,
                )[::downsample_rate]
                episode_mask_img = np.array(episode_mask_img, dtype=np.uint8)[:-1]
                assert episode_mask_img.shape[0] == xyzNrotvec[1:].shape[0]
                n, h, w, c = episode_mask_img.shape
                print(episode_mask_img.shape)
                parallel_saving(
                    group=episode_camera_data,
                    array_name="rgb",
                    shape=(n, h, w, c),
                    chunks=(1, h, w, c),
                    dtype=np.uint8,
                    overwrite=True,
                    arr_to_save=episode_mask_img,
                    max_workers=max_workers,
                    compressor=this_compressor,
                )
            # save fsr
            fsr_keys = []
            all_fsr_observations = []

            # Find all keys that match the FSR pattern
            for key in source_buffer[f"episode_{i}"].keys():
                if (
                    key.startswith("fsr_values_interp_")
                    and key.split("_")[-1].isdigit()
                ):
                    fsr_keys.append(key)
            # Sort the keys to maintain consistent order
            if len(fsr_keys) > 0:
                fsr_keys.sort(key=lambda x: int(x.split("_")[-1]))

                # Process each FSR file found
                for fsr_key in fsr_keys:
                    force_observation = source_buffer[f"episode_{i}/{fsr_key}"][:]
                    if force_adjust:
                        force_observation -= force_observation[0]
                    all_fsr_observations.append(force_observation)
                all_fsr_observations = np.stack(all_fsr_observations, axis=1)
                if force_process == "total":
                    source_buffer[f"episode_{i}/fsr_values_interp"] = (
                        compute_total_force_per_finger(all_fsr_observations)
                    )
                elif force_process == "flatten":
                    source_buffer[f"episode_{i}/fsr_values_interp"] = rearrange(
                        all_fsr_observations, "b n c -> b (n c)"
                    )

            if f"episode_{i}/fsr_values_interp" in source_buffer:
                force_observation = source_buffer[f"episode_{i}/fsr_values_interp"][:][
                    ::downsample_rate
                ]
                print(force_observation.shape)
                episode_data["fsr"] = force_observation[:-1]
                assert episode_data["fsr"].shape[0] == episode_data["pose"].shape[0]
            else:
                print(f"Path episode_{i}/fsr_values_interp not found")
            valid_episode += 1
        except Exception as e:
            print(f"Error in episode {i}:")
            print(traceback.format_exc())
            failed_episodes.append(i)
            continue

    print(f"Failed episodes: {failed_episodes}")


if __name__ == "__main__":
    main()
