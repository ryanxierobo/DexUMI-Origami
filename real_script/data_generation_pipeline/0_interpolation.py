import os

import click

from dexumi.data_recording.data_buffer import ExoDataBuffer


@click.command()
@click.option(
    "--data-dir",
    type=str,
    default="~/Dev/exoskeleton/data_local/exoskeleton_recordings",
    help="Directory path for exoskeleton recordings",
)
@click.option(
    "--target-dir",
    type=str,
    default="~/Dev/exoskeleton/data_local/exoskeleton_replay",
    help="Directory path for exoskeleton recordings",
)
@click.option(
    "--episode-indices",
    "-e",
    type=int,
    multiple=True,
    default=(0,),
    help="Episode indices to interpolate",
)
@click.option(
    "--camera-latency",
    type=float,
    default=0.1849,
    help="Camera latency in seconds",
)
@click.option(
    "--encoder-latency",
    type=float,
    default=0.25,
    # default=0.10,
    # default=0.40,
    help="encoder latency in seconds",
)
@click.option(
    "--tracking-latency",
    type=float,
    default=0.183,
    help="encoder latency in seconds",
)
@click.option(
    "--num-fsr-sources",
    type=int,
    default=3,
    help="num fsr sources",
)
@click.option(
    "--fsr-latency",
    type=float,
    default=0.23,
    help="fsr latency in seconds",
)
@click.option(
    "--enable-fsr", type=bool, default=False, help="Enable FSR processing", is_flag=True
)
@click.option("--downsample-rate", type=int, default=1, help="Downsample rate for data")
def interpolate_episodes(
    data_dir,
    target_dir,
    episode_indices,
    camera_latency,
    encoder_latency,
    tracking_latency,
    fsr_latency,
    num_fsr_sources,
    enable_fsr,
    downsample_rate,
):
    data_dir = os.path.expanduser(data_dir)
    target_dir = os.path.expanduser(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    exo_buffer = ExoDataBuffer(
        data_dir=data_dir,
        target_dir=target_dir,
        camera_latency=camera_latency,
        encoder_latency=encoder_latency,
        tracking_latency=tracking_latency,
        fsr_latencies=[fsr_latency] * num_fsr_sources,
        num_fsr_sources=num_fsr_sources,
        enable_fsr=enable_fsr,  # Pass the parameter
        downsample_rate=downsample_rate,
    )
    exo_buffer.interpolate_episode(episode_indices=list(episode_indices))


if __name__ == "__main__":
    interpolate_episodes()
