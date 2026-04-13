import glob
import os
from pathlib import Path

import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm


def preprocess(image_path, mask_path, batch, name):
    atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

    mask = np.zeros((256, 256), dtype=np.uint8)
    image = resize(imread(image_path), (256, 256))

    for label, attribute in enumerate(atts, 1):
        if Path(f"{mask_path}_{attribute}.png").exists():
            attribute_mask = imread(f"{mask_path}_{attribute}.png").max(2) >= 128
            attribute_mask = resize(attribute_mask, (256, 256))
            mask[attribute_mask] = label

    os.makedirs(f"./data/CelebAMask/preprocessed/{batch}", exist_ok=True)
    np.savez(f"./data/CelebAMask/preprocessed/{batch}/{name}", image=image.transpose(2, 0, 1), label=mask)


def main():
    images = glob.glob("./data/CelebAMask/CelebAMask-HQ/CelebA-HQ-img/*.jpg")
    for image in tqdm(images):
        name = f"{int(os.path.basename(image)[:-4]):05d}"
        batch = int(name) // 2000

        if os.path.exists(f"./data/CelebAMask/preprocessed/{batch}/{name}.npz"):
            continue

        masks = f"./data/CelebAMask/CelebAMask-HQ/CelebAMask-HQ-mask-anno/{batch}/{name}"
        preprocess(image, masks, batch, name)
    
   
if __name__ == '__main__':
    main()
