import click
import zarr

from dexumi.common.utility.matrix import (
    visualize_multiple_frames_and_points,
)


@click.command()
@click.option("-e", "--episode", type=int, default=0)
@click.option(
    "-r",
    "--replay_path",
)
def main(replay_path, episode):
    replay_buffer = zarr.open(replay_path, mode="r")
    episode_data = replay_buffer[f"episode_{episode}"]
    pose = episode_data["pose_interp"][:]
    visualize_multiple_frames_and_points(frames_dict={"pose": pose}, show_axes=True)


if __name__ == "__main__":
    main()
