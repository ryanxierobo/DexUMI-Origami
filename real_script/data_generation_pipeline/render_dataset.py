import os
import subprocess

import click
from tqdm import tqdm


def process_render_tasks(
    start,
    end,
    gpu_id,
    data_dir,
    reference_dir,
    sam2_checkpoint_path,
    save_path,
    resize_ratio,
):
    """Handle all episodes for a single GPU."""
    failed_episodes = []

    with tqdm(
        total=end - start, desc=f"GPU {gpu_id} Episodes", position=0
    ) as episode_pbar:
        for episode in range(start, end):
            # Parallel to_jpg commands
            parallel_commands_1 = [
                f"python 2_to_jpg.py -e {episode} -p 'exo' --data-dir {data_dir} --output-dir {save_path} --resize_ratio {resize_ratio}",
                f"python 2_to_jpg.py -e {episode} -p 'dex' --data-dir {data_dir} --output-dir {save_path} --resize_ratio {resize_ratio}",
            ]

            print(
                f"\nGPU {gpu_id} running to_jpg commands in parallel for episode {episode}:"
            )
            processes_1 = []
            for cmd in parallel_commands_1:
                print(f"Starting: {cmd}")
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    text=True,
                )
                processes_1.append(process)

            parallel_failed = False
            for process in processes_1:
                return_code = process.wait()
                if return_code != 0:
                    parallel_failed = True
                    break

            if parallel_failed:
                failed_episodes.append(episode)
                print(
                    f"\nParallel to_jpg processing failed for episode {episode} on GPU {gpu_id}, continuing with next episode"
                )
                episode_pbar.update(1)
                continue
            # Sequential segmentation commands
            sequential_commands = [
                f"python 3_segment.py -p 'dex' -e {episode} --data_dir {data_dir} --reference-dir {reference_dir} --sam2_checkpoint_path {sam2_checkpoint_path} --save_path {save_path}",
                f"python 3_segment.py -p 'exo' -e {episode} --data_dir {data_dir} --reference-dir {reference_dir} --sam2_checkpoint_path {sam2_checkpoint_path} --save_path {save_path}",
            ]

            print(f"\nGPU {gpu_id} running sequential commands for episode {episode}:")

            sequence_failed = False
            for cmd in sequential_commands:
                print(f"Starting: {cmd}")
                try:
                    subprocess.run(
                        cmd,
                        shell=True,
                        check=True,
                        text=True,
                    )
                except subprocess.CalledProcessError as e:
                    print(f"\nFailed command on GPU {gpu_id}: {cmd}")
                    print(f"Error output: {e.stderr}")
                    sequence_failed = True
                    break

            if sequence_failed:
                failed_episodes.append(episode)
                print(
                    f"\nSequential processing failed for episode {episode} on GPU {gpu_id}, continuing with next episode"
                )
                episode_pbar.update(1)
                continue

            # Sequential command
            inpaint_command = (
                f"python 4_inpaint_exo.py -e {episode} --buffer {data_dir}"
            )
            print(f"\nGPU {gpu_id} running: {inpaint_command}")
            try:
                process = subprocess.run(
                    inpaint_command,
                    shell=True,
                    check=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"\nFailed command on GPU {gpu_id}: {inpaint_command}")
                print(f"Error output: {e.stderr}")
                failed_episodes.append(episode)
                episode_pbar.update(1)
                continue

            # Second parallel commands (video composition)
            parallel_commands_2 = [
                f"python 5_compose_video.py -e {episode} --data-dir {data_dir}",
                f"python 5_compose_video.py -e {episode} --render-exoskeleton --data-dir {data_dir}",
                f"python 5_compose_video.py -e {episode} --render-maskout-baseline --data-dir {data_dir}",
            ]

            print(
                f"\nGPU {gpu_id} running second parallel commands for episode {episode}:"
            )
            processes_2 = []
            for cmd in parallel_commands_2:
                print(f"Starting: {cmd}")
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    text=True,
                )
                processes_2.append(process)

            parallel_failed = False
            for process in processes_2:
                return_code = process.wait()
                if return_code != 0:
                    parallel_failed = True
                    break

            if parallel_failed:
                failed_episodes.append(episode)
                print(
                    f"\nSecond parallel processing failed for episode {episode} on GPU {gpu_id}, continuing with next episode"
                )
                episode_pbar.update(1)
                continue

            episode_pbar.update(1)

    return failed_episodes


@click.command()
@click.option("--start", required=True, type=int, help="Starting episode number")
@click.option("--end", required=True, type=int, help="Ending episode number")
@click.option("--gpu_id", required=True, type=int, help="GPU ID to use")
@click.option("--data_dir", required=True, help="Data directory")
@click.option("--reference_dir", required=True, help="Reference directory")
@click.option("--sam2_checkpoint_path", required=True, help="SAM2 checkpoint path")
@click.option("--save_path", required=True, help="Save path")
@click.option(
    "--resize_ratio",
    default=1.0,
    type=float,
)
def main(
    start,
    end,
    gpu_id,
    data_dir,
    reference_dir,
    sam2_checkpoint_path,
    save_path,
    resize_ratio,
):
    """Process render tasks for episodes from START to END on GPU_ID."""
    print(f"Processing episodes {start} to {end} on GPU {gpu_id}")
    failed_episodes = process_render_tasks(
        start,
        end,
        gpu_id,
        data_dir,
        reference_dir,
        sam2_checkpoint_path,
        save_path,
        resize_ratio,
    )
    print(f"\nFailed episodes on GPU {gpu_id}:", failed_episodes)


if __name__ == "__main__":
    main()
