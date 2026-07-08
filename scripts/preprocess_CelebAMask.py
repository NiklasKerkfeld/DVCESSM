import os
import glob
import numpy as np
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

ATTRIBUTES = [
    'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
    'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat'
]


def process_single_item(image_path):
    """Worker function for parallel processing and tracking channel statistics (0-255 range)"""
    name = f"{int(os.path.basename(image_path)[:-4]):05d}"
    batch = int(name) // 2000
    out_dir = Path(f"./data/CelebAMask/preprocessed/{batch}")
    out_path = out_dir / f"{name}.npz"

    # If already preprocessed, load the image to extract its statistics
    if out_path.exists():
        with np.load(out_path) as data:
            # Map back from (C, H, W) to (H, W, C)
            img_spatial = data['image'].transpose(1, 2, 0)
    else:
        img = imread(image_path)
        img_spatial = (resize(img, (256, 256), preserve_range=True)).astype(np.uint8)

        mask_dir = Path(f"./data/CelebAMask/CelebAMask-HQ/CelebAMask-HQ-mask-anno/{batch}")
        combined_mask = np.zeros((256, 256), dtype=np.uint8)

        for label, attr in enumerate(ATTRIBUTES, 1):
            attr_path = mask_dir / f"{name}_{attr}.png"
            if attr_path.exists():
                attr_img = imread(str(attr_path))
                attr_img = attr_img.max(axis=2)

                attr_mask = resize(attr_img >= 128, (256, 256), order=0, preserve_range=True,
                                   anti_aliasing=False)
                combined_mask[attr_mask > 0] = label

        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, image=img_spatial.transpose(2, 0, 1), label=combined_mask)

    # Convert to float64 to prevent integer overflow during summation
    img_float = img_spatial.astype(np.float64)

    # Sum elements across Height and Width axes in the 0-255 space
    channel_sum = img_float.sum(axis=(0, 1))
    channel_sum_sq = (img_float ** 2).sum(axis=(0, 1))

    return channel_sum, channel_sum_sq


def main():
    images = glob.glob("./data/CelebAMask/CelebAMask-HQ/CelebA-HQ-img/*.jpg")

    total_images = len(images)
    if total_images == 0:
        print("No images found.")
        return

    print("Processing images and gathering statistics...")
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_single_item, images), total=total_images))

    # Initialize aggregators
    total_sum = np.zeros(3, dtype=np.float64)
    total_sum_sq = np.zeros(3, dtype=np.float64)

    # Total number of pixels per channel across the entire dataset
    total_pixels = total_images * 256 * 256

    # Accumulate results from all workers
    for c_sum, c_sum_sq in results:
        total_sum += c_sum
        total_sum_sq += c_sum_sq

    # Compute global Mean and Standard Deviation in 0-255 space
    mean = total_sum / total_pixels
    variance = (total_sum_sq / total_pixels) - (mean ** 2)
    std = np.sqrt(variance)

    print("\n--- Dataset Statistics (Raw 0 to 255 Space) ---")
    print(f"Mean (R, G, B): {mean}")
    print(f"Std  (R, G, B): {std}")


if __name__ == "__main__":
    main()