import glob
import os
from pathlib import Path

import numpy as np
from skimage.io import imread
from skimage.transform import resize


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
    np.savez(f"./data/CelebAMask/preprocessed/{batch}/{name}", image=image, label=mask)


def main():
    images = glob.glob("./data/CelebAMask/CelebAMask-HQ/CelebA-HQ-img/*.jpg")
    for image in images[:100]:
        name = f"{int(os.path.basename(image)[:-4]):05d}"
        batch = int(name) // 2000
        masks = f"./data/CelebAMask/CelebAMask-HQ/CelebAMask-HQ-mask-anno/{batch}/{name}"
        preprocess(image, masks, batch, name)
    
   
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    main()

    data = np.load("data/CelebAMask/preprocessed/5/11045.npz")
    image = data["image"]
    mask = data["label"]
    print(image.shape, mask.shape)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")


    print(f"{mask.min()}, {mask.max()}, {np.unique(mask)}")
    ax[1].imshow(mask, cmap='tab20', vmin=0, vmax=20)
    ax[1].set_title("Segmentation Mask")
    ax[1].axis("off")

    plt.show()

    plt.imshow(mask, cmap='tab20', vmin=0, vmax=20)
    plt.show()