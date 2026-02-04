import glob
import os

import click
import cv2
import imageio
import numpy as np
import zarr
from matplotlib import pyplot as plt

from dexumi.common.utility.video import extract_frames_videos


def compose_masked_images(image1, mask1, image2, mask2):
    assert image1.shape[:2] == mask1.shape, "Image 1 and mask 1 shapes don't match"
    assert image2.shape[:2] == mask2.shape, "Image 2 and mask 2 shapes don't match"
    assert image1.shape == image2.shape, "Images must have the same dimensions"

    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    # result = np.zeros_like(image1)
    result = np.ones_like(image1)
    result[:, :, 0] = 255
    result[mask1] = image1[mask1]
    result[mask2] = image2[mask2]
    return result


@click.command()
@click.option(
    "--data-dir",
    default="~/Dev/exoskeleton/data_local/exoskeleton_replay",
    help="Path to the inpainted video file",
)
@click.option("--episode", "-e", default=0, type=int, help="Episode number")
@click.option(
    "--render-exoskeleton", is_flag=True, help="Render exoskeleton in the output video"
)
@click.option(
    "--skip-seg-index", default=1, type=int, help="Index to start skipping segments"
)
@click.option(
    "--render-maskout-baseline", is_flag=True, help="Render maskout baseline video"
)
def main(
    data_dir,
    episode,
    render_exoskeleton,
    skip_seg_index,
    render_maskout_baseline,
):
    # Expand user paths
    data_dir = os.path.expanduser(data_dir)
    inpainted_video = os.path.join(
        data_dir, f"episode_{episode}", "inpainted", "exo_img", "inpaint_out.mp4"
    )
    exo_seg_mask_dir = os.path.join(data_dir, f"episode_{episode}", "exo_seg_mask")
    dex_video = os.path.join(data_dir, f"episode_{episode}", "dex_camera_0.mp4")
    dex_seg_mask_dir = os.path.join(data_dir, f"episode_{episode}", "dex_seg_mask")
    # Convert boolean mask to integer indices using numpy where
    exo_inpainted_frame = extract_frames_videos(inpainted_video)

    exo_mask_files = glob.glob(os.path.join(exo_seg_mask_dir, "*.png"))
    exo_mask_files.sort(
        key=lambda x: int("".join(filter(str.isdigit, os.path.basename(x))))
    )
    exo_seg_masks = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in exo_mask_files]
    dex_hand_image = extract_frames_videos(dex_video)[skip_seg_index:]
    dex_seg_mask_files = glob.glob(os.path.join(dex_seg_mask_dir, "*.png"))
    dex_seg_mask_files.sort(
        key=lambda x: int("".join(filter(str.isdigit, os.path.basename(x))))
    )
    dex_seg_mask_files = dex_seg_mask_files[skip_seg_index:]
    dex_seg_masks = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in dex_seg_mask_files]

    height, width = exo_inpainted_frame[0].shape[:2]

    # Prepare frames for video creation
    frames = []
    print("dex masks shape:", len(dex_seg_masks))
    print("exo_masks shape:", len(exo_seg_masks))
    print("exo_inpainted_frame shape:", len(exo_inpainted_frame))
    for i in range(len(exo_inpainted_frame)):
        # Resize everything to match inpainted frame size
        this_exo_seg_masks = cv2.resize(exo_seg_masks[i], (width, height))
        this_dex_seg_masks = cv2.resize(dex_seg_masks[i], (width, height))
        this_dex_frame = cv2.resize(dex_hand_image[i], (width, height))

        if render_exoskeleton:
            dex_mask = this_dex_seg_masks
            exo_mask = ~this_exo_seg_masks
            output_path = os.path.join(
                data_dir, f"episode_{episode}", "debug_combined.mp4"
            )
        elif render_maskout_baseline:
            dex_mask = np.zeros_like(this_dex_seg_masks)
            exo_mask = ~this_exo_seg_masks
            output_path = os.path.join(
                data_dir, f"episode_{episode}", "maskout_baseline.mp4"
            )
        else:
            dex_mask = this_dex_seg_masks & this_exo_seg_masks
            exo_mask = np.ones_like(this_exo_seg_masks)
            output_path = os.path.join(data_dir, f"episode_{episode}", "combined.mp4")
        composed_frame = compose_masked_images(
            exo_inpainted_frame[i], exo_mask, this_dex_frame, dex_mask
        )
        composed_frame = cv2.cvtColor(composed_frame, cv2.COLOR_BGR2RGB)
        frames.append(composed_frame.astype(np.uint8))

    # Write video using imageio
    imageio.mimwrite(output_path, frames, fps=30, quality=8)
    print(f"Video saved to {output_path}")


if __name__ == "__main__":
    main()
