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
    """Worker function for parallel processing"""
    name = f"{int(os.path.basename(image_path)[:-4]):05d}"
    batch = int(name) // 2000
    out_dir = Path(f"./data/CelebAMask/preprocessed/{batch}")
    out_path = out_dir / f"{name}.npz"

    if out_path.exists():
        return

    img = imread(image_path)
    img_resized = (resize(img, (256, 256), preserve_range=True)).astype(np.uint8)

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
    np.savez_compressed(out_path, image=img_resized.transpose(2, 0, 1), label=combined_mask)


def main():
    images = glob.glob("./data/CelebAMask/CelebAMask-HQ/CelebA-HQ-img/*.jpg")

    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_single_item, images), total=len(images)))


if __name__ == "__main__":
    main()