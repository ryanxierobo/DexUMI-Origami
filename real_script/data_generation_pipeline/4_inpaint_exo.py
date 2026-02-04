import os
from pathlib import Path

import click
from dexumi.constants import INPAINT_RESIZE_RATIO


@click.command()
@click.option(
    "--buffer",
    default="../../data_local/exoskeleton_replay",
    type=str,
    help="Path to the replay buffer directory",
)
@click.option("--episode", "-e", required=True, type=int, help="Episode number")
def inpaint_video(buffer, episode):
    # Convert to absolute Path object for easier manipulation
    buffer_path = Path(buffer).resolve()  # Convert to absolute path
    episode_dir = buffer_path / f"episode_{episode}"

    # Validate episode directory exists
    if not episode_dir.exists():
        raise click.BadParameter(f"Episode directory {episode_dir} does not exist")

    # Construct absolute paths
    video_path = str(episode_dir / "exo_img")
    mask_path = str(episode_dir / "exo_seg_mask")
    output_path = str(episode_dir / "inpainted")

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Change working directory to ProPainter
    os.chdir("../../../ProPainter/")

    # Command to execute with absolute paths
    command = (
        f"python inference_propainter.py "
        f"--video {video_path} "
        f"--mask {mask_path} "
        f"--output {output_path} "
        f"--resize_ratio {INPAINT_RESIZE_RATIO} "
    )

    # Execute the command
    os.system(command)


if __name__ == "__main__":
    inpaint_video()
