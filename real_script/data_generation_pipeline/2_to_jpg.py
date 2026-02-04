import os

import click
import cv2
import ffmpeg
import numpy as np
import zarr
from tqdm import tqdm


@click.command()
@click.option(
    "--data-dir",
    "-d",
    default="~/Dev/exoskeleton/data_local/exoskeleton_recordings",
    required=True,
    help="Path to input video file",
)
@click.option(
    "--output-dir",
    "-o",
    default="~/Dev/exoskeleton/data_local/exoskeleton_replay",
    required=True,
    help="Directory for output JPG sequences",
)
@click.option(
    "--episode",
    "-e",
    default=0,
    help="Episode number to process",
    type=int,
)
@click.option(
    "--quality",
    "-q",
    default=2,
    help="JPEG quality (2-31, lower is better quality)",
    type=int,
)
@click.option(
    "--prefix",
    "-p",
    default="exo",
    help="Prefix for the output JPG filenames",
    type=str,
)
@click.option(
    "--resize_ratio",
    "-r",
    default=1.0,
    help="Ratio to resize the frames (e.g., 0.8 for 80% of original size)",
    type=float,
)
@click.option(
    "--method",
    "-m",
    type=click.Choice(["ffmpeg", "opencv"]),
    default="opencv",
    help="Method to use for frame extraction",
)
def convert_episode_to_jpg(
    data_dir, episode, output_dir, prefix, quality, resize_ratio, method
):
    """Convert episode video to sequence of JPG images."""
    assert prefix in ["exo", "dex"], "Prefix must be either 'exo' or 'dex'"

    data_dir = os.path.expanduser(data_dir)
    output_dir = os.path.expanduser(output_dir)
    data_buffer = zarr.open(data_dir, mode="a")
    episode_data = data_buffer[f"episode_{episode}"]
    valid_indices = episode_data["valid_indices"][:]
    input_video = os.path.join(data_dir, f"episode_{episode}", f"{prefix}_camera_0.mp4")
    episode_output_dir = os.path.join(output_dir, f"episode_{episode}", f"{prefix}_img")
    os.makedirs(episode_output_dir, exist_ok=True)
    if prefix == "dex":
        # the replay is already with valid timestamps
        indices = None
    else:
        indices = valid_indices
    if method == "ffmpeg":
        convert_video_to_jpg_ffmpeg(
            input_video, episode_output_dir, quality, resize_ratio, indices
        )
    else:
        convert_video_to_jpg_opencv(
            input_video, episode_output_dir, quality, resize_ratio, indices
        )


def convert_video_to_jpg_ffmpeg(
    input_video, output_directory, quality=2, resize_ratio=None, valid_indices=None
):
    """
    Convert video file to sequence of JPG images using ffmpeg.

    Args:
        input_video (str): Path to input video file
        output_directory (str): Directory to save output JPG sequences
        quality (int): JPEG quality (2-31, lower is better quality)
        resize_ratio (float, optional): Ratio to resize the frames
        valid_indices (arr, optional):  boolean indices
    """
    os.makedirs(output_directory, exist_ok=True)
    output_pattern = os.path.join(output_directory, "%05d.jpg")

    try:
        # Get video information
        probe = ffmpeg.probe(input_video)
        video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
        width = int(video_info["width"])
        height = int(video_info["height"])

        # Setup ffmpeg stream
        stream = ffmpeg.input(input_video)

        # Add select filter based on valid indices
        if valid_indices is not None:
            indices_str = "+".join(
                [f"eq(n,{i})" for i, valid in enumerate(valid_indices) if valid]
            )
            stream = ffmpeg.filter(stream, "select", indices_str)
            stream = ffmpeg.filter(stream, "setpts", "N/TB")

        # Add resize filter if resize_ratio is provided
        if resize_ratio is not None:
            new_width = int(width * resize_ratio)
            new_height = int(height * resize_ratio)
            stream = ffmpeg.filter(stream, "scale", new_width, new_height)

        # Setup output with quality setting
        stream = ffmpeg.output(
            stream, output_pattern, **{"q:v": quality, "start_number": 0}
        )

        # Run the ffmpeg command
        ffmpeg.run(stream, overwrite_output=True)
        print(f"Successfully converted {input_video} to JPG sequences")

    except ffmpeg.Error as e:
        print(f"An error occurred: {e.stderr.decode()}")


def convert_video_to_jpg_opencv(
    input_video, output_directory, quality=2, resize_ratio=None, valid_indices=None
):
    """
    Convert video file to sequence of JPG images using OpenCV.

    Args:
        input_video (str): Path to input video file
        output_directory (str): Directory to save output JPG sequences
        quality (int): JPEG quality (2-31, lower is better quality)
        resize_ratio (float, optional): Ratio to resize the frames
        valid_indices (arr, optional):  boolean indices
    """
    os.makedirs(output_directory, exist_ok=True)
    # Open video file
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video}")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate new dimensions if resize_ratio is provided
    if resize_ratio is not None:
        new_width = int(width * resize_ratio)
        new_height = int(height * resize_ratio)

    # Convert quality from ffmpeg scale (2-31) to OpenCV scale (0-100)
    opencv_quality = int((1 - (quality - 2) / 29) * 100)

    # Process frames
    output_frame_count = 0
    print("Processing frames...")
    for frame_idx in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        # Check if we should keep this frame
        if valid_indices is not None and not valid_indices[frame_idx]:
            continue

        # Resize if needed
        if resize_ratio is not None:
            frame = cv2.resize(frame, (new_width, new_height))

        # Save frame
        output_path = os.path.join(output_directory, f"{output_frame_count:05d}.jpg")
        cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, opencv_quality])
        output_frame_count += 1

    cap.release()
    print(f"Successfully converted {input_video} to {output_frame_count} JPG sequences")


if __name__ == "__main__":
    convert_episode_to_jpg()
