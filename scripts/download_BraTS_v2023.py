import glob
from pathlib import Path
import os

import numpy as np
import synapseclient
import zipfile

import torch
from matplotlib import pyplot as plt
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, SpatialPadd, \
    CenterSpatialCropd, ClipIntensityPercentilesd, NormalizeIntensityd, CastToTyped, ConcatItemsd, \
    CropForegroundd
from tqdm import tqdm

KEYS = ["T1", "T1ce", "T2", "FLAIR"]
SEGMENTATION = ["label"]

# Define the transforms
transforms = Compose([
    LoadImaged(keys=KEYS + SEGMENTATION, allow_missing_keys=True),
    EnsureChannelFirstd(keys=KEYS + SEGMENTATION, allow_missing_keys=True),

    # crop center
    SpatialPadd(keys=KEYS + SEGMENTATION, spatial_size=(128, 128, -1,),
                allow_missing_keys=True),
    CenterSpatialCropd(keys=KEYS + SEGMENTATION, roi_size=(128, 128, -1,),
                       allow_missing_keys=True),

    ConcatItemsd(keys=KEYS, name="image"),
    CropForegroundd(
        keys=["image", "label"],
        source_key="image",
        select_fn=lambda x: x > 0,  # Or your specific threshold
        channel_indices=1,
        margin=0
    ),

    # scale intensities
    ClipIntensityPercentilesd(
        keys=KEYS,
        lower=1, upper=99, channel_wise=True),
    NormalizeIntensityd(keys=KEYS, nonzero=True, channel_wise=True),

    # ensure right format
    CastToTyped(keys=["image", "label"], dtype=(torch.float32, torch.uint8))
])


def preprocess():
    for sample in tqdm(glob.glob('./data/BraTS/V2024/training_data1_v2/*'), desc="preprocessing"):
        print(sample)
        name = sample.split('/')[-1]

        volume_dir = f"./data/BraTS/V2024/preprocessed/{name}"
        os.makedirs(volume_dir, exist_ok=True)

        data = transforms({
            "T1": f"{sample}/{name}-t1n.nii.gz",
            "T1ce": f"{sample}/{name}-t1c.nii.gz",
            "T2": f"{sample}/{name}-t2w.nii.gz",
            "FLAIR": f"{sample}/{name}-t2f.nii.gz",
            "label": f"{sample}/{name}-seg.nii.gz"
        })
        image = data['image']
        label = data['label']

        num_slices = image.shape[1]

        for s in range(num_slices):
            slice_image = image[..., s]
            slice_label = label[..., s]

            save_path = os.path.join(volume_dir, f"slice{s:03d}.npz")

            np.savez(
                save_path,
                image=slice_image,
                label=slice_label,
            )


def main():
    print(f"current working directory: {os.getcwd()}")

    syn = synapseclient.Synapse()
    syn.login()

    filepath = Path('./data/BraTS/BraTS2024-BraTS-GLI-TrainingData.zip')
    if not filepath.is_file():
        syn.get(entity='syn60086071', downloadLocation='./data/BraTS')

        print(f"Download complete!")

    if not os.path.isdir('./data/BraTS/V2024'):
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall('./data/BraTS/V2024')

        print("Extraction complete!")

    preprocess()


def visualize_npz_slice(npz_path):
    # Load the data
    data = np.load(npz_path)
    image = data['image']  # Shape: [4, H, W]
    label = data['label']  # Shape: [1, H, W]

    num_channels = image.shape[0]
    fig, axes = plt.subplots(1, num_channels + 1, figsize=(15, 5))

    # Names for BraTS modalities
    modality_names = ["T1", "T1ce", "T2", "FLAIR"]

    # Plot each channel
    for i in range(num_channels):
        axes[i].imshow(image[i], cmap='gray')
        axes[i].set_title(modality_names[i] if i < len(modality_names) else f"Ch {i}")
        axes[i].axis('off')

    # Plot Label Overlay (on the last modality, usually FLAIR)
    axes[-1].imshow(label[0], cmap='gray')
    axes[-1].set_title("Label")
    axes[-1].axis('off')

    plt.suptitle(f"File: {os.path.basename(npz_path)}")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
