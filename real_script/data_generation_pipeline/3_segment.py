import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

from dexumi.common.utility.file import read_pickle


def save_mask(mask, path, format, quality=95):
    imageio.imwrite(path, mask, quality=quality if format == "jpg" else None)


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def convert_to_binary_mask_3channel(mask_array):
    """
    Convert a NumPy array to 3-channel binary mask images.

    Parameters:
    mask_array: numpy.ndarray of shape (N, H, W) or (N, H, W, 1)
        Input mask array

    Returns:
    numpy.ndarray
        Binary mask array with values 0 or 255 (uint8 type) with 3 channels
    """
    # Ensure proper dimensions
    if mask_array.ndim == 4:
        mask_array = mask_array.squeeze(-1)

    # Convert to binary (0 or 1)
    binary_mask = (mask_array > 0).astype(np.uint8)

    # Scale to 0 or 255 for better visibility
    binary_mask = binary_mask * 255

    # Create 3-channel masks
    masks_3channel = np.stack([binary_mask] * 3, axis=-1)

    return masks_3channel


@click.command()
@click.option(
    "--data_dir",
    required=True,
    default="~/Dev/DexUMI/data_local/exoskeleton_replay",
    help="Directory of JPEG frames.",
)
@click.option(
    "--reference-dir",
    type=click.Path(),
    default="~/Dev/DexUMI/data_local/reference_data",
    help="Base directory for reference data",
)
@click.option("--episode_id", "-e", required=True, type=int, help="Episode ID.")
@click.option(
    "--sam2_checkpoint_path",
    required=True,
    default="~/Dev/sam2/checkpoints/sam2.1_hiera_large.pt",
    help="Path to SAM checkpoint.",
)
@click.option(
    "--save_path",
    required=True,
    default="~/Dev/DexUMI/data_local/exoskeleton_replay",
    type=click.Path(),
    help="Path to save the segmentation results (.npy file).",
)
@click.option(
    "--prefix",
    "-p",
    default="exo",
    help="Prefix for the output JPG filenames",
    type=str,
)
@click.option(
    "--format",
    "-f",
    default="png",
    type=click.Choice(["jpg", "png"]),
    help="Output format for binary masks (jpg or png)",
)
@click.option(
    "--manual_annotation",
    "-m",
    is_flag=True,
    default=False,
    help="Enable manual annotation mode",
)
@click.option(
    "--viz_mask",
    "-v",
    is_flag=True,
    default=False,
    help="Enable visualization of masks",
)
@click.option(
    "--max_workers",
    default=32,
    type=int,
    help="Maximum number of workers for ThreadPoolExecutor",
)
def main(
    data_dir,
    reference_dir,
    episode_id,
    sam2_checkpoint_path,
    save_path,
    prefix,
    format,
    manual_annotation,
    viz_mask,
    max_workers,
):
    data_dir = os.path.expanduser(data_dir)
    reference_dir = os.path.expanduser(reference_dir)
    episode_dir = os.path.join(data_dir, f"episode_{episode_id}")
    episode_jpg_dir = os.path.join(episode_dir, f"{prefix}_img")
    save_path = os.path.expanduser(save_path)
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    sam2_checkpoint_path = os.path.expanduser(sam2_checkpoint_path)
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(
        model_cfg, sam2_checkpoint_path, device=device
    )

    # scan all the JPEG frame names in this directory
    frame_names = [
        p
        for p in os.listdir(episode_jpg_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    t1 = time.time()
    inference_state = predictor.init_state(video_path=episode_jpg_dir)
    predictor.reset_state(inference_state)
    t2 = time.time()
    print(f"init_state took {t2 - t1:.2f} seconds")

    ann_frame_idx = 0  # the frame index we interact with

    if manual_annotation:
        points_list = []
        labels_list = []
        ann_obj_id = 1
        frame_idx = 0
        plt.figure(figsize=(9, 6))
        plt.title(f"frame {frame_idx}")
        plt.imshow(Image.open(os.path.join(episode_jpg_dir, frame_names[frame_idx])))

        def onclick(event):
            if event.button == 1:  # Left click for positive points
                points_list.append([event.xdata, event.ydata])
                labels_list.append(1)
                plt.plot(event.xdata, event.ydata, "g*", markersize=10)
            elif event.button == 3:  # Right click for negative points
                points_list.append([event.xdata, event.ydata])
                labels_list.append(0)
                plt.plot(event.xdata, event.ydata, "r*", markersize=10)
            plt.draw()

        # Connect the click event
        cid = plt.gcf().canvas.mpl_connect("button_press_event", onclick)
        plt.show()
        # Convert collected points to numpy arrays
        points = np.array(points_list, dtype=np.float32)
        labels = np.array(labels_list, np.int32)
        # Save points and labels to a pickle file
        points_data = {"points": points, "labels": labels}
        #########################################################################################
        # change the directory to save the points for different part of hand
        points_pkl_path = f"{save_path}/{prefix}_finger_points.pkl"
        with open(points_pkl_path, "wb") as f:
            pickle.dump(points_data, f)
        #########################################################################################

        # Only proceed if points were collected
        if len(points_list) > 0:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points,
                labels=labels,
            )

            plt.figure(figsize=(9, 6))
            plt.title(f"frame {ann_frame_idx}")
            plt.imshow(
                Image.open(os.path.join(episode_jpg_dir, frame_names[ann_frame_idx]))
            )
            show_points(points, labels, plt.gca())
            show_mask(
                (out_mask_logits[0] > 0.0).cpu().numpy(),
                plt.gca(),
                obj_id=out_obj_ids[0],
            )
            plt.show()
        exit()
    else:
        reference_thumb_points = read_pickle(
            file_name=os.path.join(reference_dir, f"{prefix}_thumb_points.pkl")
        )
        reference_finger_points = read_pickle(
            file_name=os.path.join(reference_dir, f"{prefix}_finger_points.pkl")
        )
        if prefix == "exo":
            try:
                reference_pinky_points = read_pickle(
                    file_name=os.path.join(reference_dir, f"{prefix}_pinky_points.pkl")
                )
            except Exception as e:
                print(f"Error processing pinky points: {e}")
                print(
                    "Pinky points not found in reference data. Skipping pinky segmentation."
                )
        ann_obj_id = 1
        finger_points = reference_finger_points["points"]
        finger_labels = reference_finger_points["labels"]
        print(
            f"finger_points: {finger_points.shape}",
            f"finger_labels: {finger_labels.shape}",
        )
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=finger_points,
            labels=finger_labels,
        )
        if viz_mask:
            plt.figure(figsize=(9, 6))
            plt.title(f"frame {ann_frame_idx}")
            plt.imshow(
                Image.open(os.path.join(episode_jpg_dir, frame_names[ann_frame_idx]))
            )
            show_points(finger_points, finger_labels, plt.gca())
            for i, out_obj_id in enumerate(out_obj_ids):
                show_mask(
                    (out_mask_logits[i] > 0.0).cpu().numpy(),
                    plt.gca(),
                    obj_id=out_obj_id,
                )
            plt.show()

        ann_obj_id = 2
        thumb_points = reference_thumb_points["points"]
        thumb_labels = reference_thumb_points["labels"]

        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=thumb_points,
            labels=thumb_labels,
        )
        if viz_mask:
            plt.figure(figsize=(9, 6))
            plt.title(f"frame {ann_frame_idx}")
            plt.imshow(
                Image.open(os.path.join(episode_jpg_dir, frame_names[ann_frame_idx]))
            )
            show_points(thumb_points, thumb_labels, plt.gca())
            for i, out_obj_id in enumerate(out_obj_ids):
                show_mask(
                    (out_mask_logits[i] > 0.0).cpu().numpy(),
                    plt.gca(),
                    obj_id=out_obj_id,
                )
            plt.show()
        if prefix == "exo":
            try:
                ann_obj_id = 3
                pinky_points = reference_pinky_points["points"]
                pinky_labels = reference_pinky_points["labels"]

                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    points=pinky_points,
                    labels=pinky_labels,
                )
                if viz_mask:
                    plt.figure(figsize=(9, 6))
                    plt.title(f"frame {ann_frame_idx}")
                    plt.imshow(
                        Image.open(
                            os.path.join(episode_jpg_dir, frame_names[ann_frame_idx])
                        )
                    )
                    show_points(pinky_points, pinky_labels, plt.gca())
                    for i, out_obj_id in enumerate(out_obj_ids):
                        show_mask(
                            (out_mask_logits[i] > 0.0).cpu().numpy(),
                            plt.gca(),
                            obj_id=out_obj_id,
                        )
                    plt.show()
            except Exception as e:
                print(f"Error processing pinky points: {e}")
                print(
                    "Pinky points not found in reference data. Skipping pinky segmentation."
                )
    # run propagation throughout the video and collect the results in a dict
    t3 = time.time()
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    t4 = time.time()
    print(f"propagate_in_video took {t4 - t3:.2f} seconds")
    # first union frame
    t5 = time.time()
    out_frames_to_save = []
    finger_frames_to_save = []
    thumb_frames_to_save = []

    for out_frame_idx in range(0, len(frame_names)):
        # Initialize combined mask for this frame
        combined_mask = np.zeros_like(list(video_segments[out_frame_idx].values())[0])[
            0
        ]

        # Process each object mask in the current frame
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            # Update combined mask
            combined_mask = np.logical_or(combined_mask, out_mask[0])

            # Save finger mask (object ID 1)
            if out_obj_id == 1:
                finger_frames_to_save.append(out_mask[0])
            # Save thumb mask (object ID 2)
            elif out_obj_id == 2:
                thumb_frames_to_save.append(out_mask[0])

        # Save combined mask for this frame
        out_frames_to_save.append(combined_mask)

    # Convert lists to numpy arrays
    out_frames_to_save = np.array(out_frames_to_save)
    finger_frames_to_save = np.array(finger_frames_to_save)
    thumb_frames_to_save = np.array(thumb_frames_to_save)
    t_6 = time.time()
    print(f"extract mask took {t_6 - t5:.2f} seconds")

    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_img_dir = os.path.join(
        save_path, f"episode_{episode_id}", f"{prefix}_seg_mask"
    )
    save_finger_img_dir = os.path.join(
        save_path, f"episode_{episode_id}", f"{prefix}_finger_seg_mask"
    )
    save_thumb_img_dir = os.path.join(
        save_path, f"episode_{episode_id}", f"{prefix}_thumb_seg_mask"
    )
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_finger_img_dir, exist_ok=True)
    os.makedirs(save_thumb_img_dir, exist_ok=True)
    t_44 = time.time()
    binary_masks = convert_to_binary_mask_3channel(out_frames_to_save)
    binary_finger_masks = convert_to_binary_mask_3channel(finger_frames_to_save)
    binary_thumb_masks = convert_to_binary_mask_3channel(thumb_frames_to_save)
    t5 = time.time()
    print(f"convert_to_binary_mask_3channel took {t5 - t_44:.2f} seconds")

    # Use ThreadPoolExecutor with a maximum number of workers
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        # Submit regular masks
        for i, mask in enumerate(binary_masks):
            path = f"{save_img_dir}/{i:05d}.{format}"
            futures.append(executor.submit(save_mask, mask, path, format))

        # Submit finger masks
        for i, mask in enumerate(binary_finger_masks):
            path = f"{save_finger_img_dir}/{i:05d}.{format}"
            futures.append(executor.submit(save_mask, mask, path, format))

        # Submit thumb masks
        for i, mask in enumerate(binary_thumb_masks):
            path = f"{save_thumb_img_dir}/{i:05d}.{format}"
            futures.append(executor.submit(save_mask, mask, path, format))

        # Wait for all tasks to complete
        for future in as_completed(futures):
            future.result()  # This will raise any exceptions that occurred

    t6 = time.time()
    print(f"save masks took {t6 - t5:.2f} seconds")

    # render 10 evenly spaced frames
    plt.close("all")
    if viz_mask or manual_annotation:
        total_frames = len(frame_names)
        evenly_spaced_frame_indices = np.linspace(0, total_frames - 1, 10, dtype=int)
        plt.figure(figsize=(12, 8))
        for i, out_frame_idx in enumerate(evenly_spaced_frame_indices):
            plt.subplot(2, 5, i + 1)
            plt.title(f"frame {out_frame_idx}")
            plt.imshow(
                Image.open(os.path.join(episode_jpg_dir, frame_names[out_frame_idx]))
            )
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
            plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.show()


if __name__ == "__main__":
    main()
