import os

import cv2
import ffmpeg


def extract_frames_videos(video_path: str, BGR2RGB=False, resize_ratio=None):
    """
    Extract frames from a video with optional BGR to RGB conversion and resizing

    Parameters:
    video_path (str): Path to the video file
    BGR2RGB (bool): Convert BGR to RGB if True
    resize_ratio (float): Ratio to resize frames (e.g., 0.5 for half size, 2 for double size)
                         None means no resizing

    Returns:
    list: List of extracted frames (optionally processed)
    """
    frames = []
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Apply BGR to RGB conversion if requested
            if BGR2RGB:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply resizing if ratio is provided
            if resize_ratio is not None and resize_ratio > 0:
                new_width = int(frame.shape[1] * resize_ratio)
                new_height = int(frame.shape[0] * resize_ratio)
                frame = cv2.resize(
                    frame,
                    (new_width, new_height),
                    interpolation=cv2.INTER_AREA
                    if resize_ratio < 1
                    else cv2.INTER_LINEAR,
                )

            frames.append(frame)
        else:
            break

    cap.release()
    return frames


def load_images_from_folder(folder_path, BGR2RGB=False, resize_ratio=None):
    """
    Load images from a folder with optional BGR to RGB conversion and resizing

    Parameters:
    folder_path (str): Path to the folder containing images
    BGR2RGB (bool): Convert BGR to RGB if True
    resize_ratio (float): Ratio to resize images (e.g., 0.5 for half size, 2 for double size)
                         None means no resizing

    Returns:
    list: List of loaded (and optionally processed) images
    """
    # Get list of files and sort them numerically
    images = []
    files = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]
    # Sort files based on the numeric part
    sorted_files = sorted(files, key=lambda x: int(x.replace(".jpg", "")))

    for filename in sorted_files:
        img_path = os.path.join(folder_path, filename)
        # Read image using cv2
        img = cv2.imread(img_path)
        if img is not None:
            if BGR2RGB:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Apply resizing if ratio is provided
            if resize_ratio is not None and resize_ratio > 0:
                new_width = int(img.shape[1] * resize_ratio)
                new_height = int(img.shape[0] * resize_ratio)
                img = cv2.resize(
                    img,
                    (new_width, new_height),
                    interpolation=cv2.INTER_AREA
                    if resize_ratio < 1
                    else cv2.INTER_LINEAR,
                )

            images.append(img)

    return images


def convert_video_to_images(
    input_video, output_directory, format="jpg", quality=2, start_number=0
):
    """Convert video file to sequence of images (JPG/PNG) using ffmpeg.

    Args:
        input_video: Path to input video file
        output_directory: Directory to save output images
        format: Output format ('jpg' or 'png')
        quality: Image quality (1-31, lower is better)
        start_number: Starting number for image sequence
    """
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Validate format
    format = format.lower()
    if format not in ["jpg", "png"]:
        raise ValueError("Format must be 'jpg' or 'png'")

    # Construct output pattern
    output_pattern = os.path.join(output_directory, f"%05d.{format}")

    try:
        # Setup ffmpeg stream
        stream = ffmpeg.input(input_video)

        # Configure output based on format
        output_args = {"start_number": start_number}
        if format == "jpg":
            output_args["q:v"] = quality

        stream = ffmpeg.output(stream, output_pattern, **output_args)

        # Run the ffmpeg command
        ffmpeg.run(stream, overwrite_output=True)
        print(f"Successfully converted {input_video} to {format.upper()} sequences")

    except ffmpeg.Error as e:
        print(f"An error occurred: {e.stderr.decode()}")
