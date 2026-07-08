import glob
import os
import zipfile
from pathlib import Path

import numpy as np
import synapseclient
import torch
from matplotlib import pyplot as plt
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, SpatialPadd, \
    CenterSpatialCropd, ClipIntensityPercentilesd, CastToTyped, ConcatItemsd, \
    CropForegroundd
from synapseclient.operations import get, FileOptions
from tqdm import tqdm

KEYS = ["T1", "T1ce", "T2", "FLAIR"]
SEGMENTATION = ["label"]

# Define the transforms
transforms = Compose([
    LoadImaged(keys=KEYS + SEGMENTATION, allow_missing_keys=True),
    EnsureChannelFirstd(keys=KEYS + SEGMENTATION, allow_missing_keys=True),

    ConcatItemsd(keys=KEYS, name="image"),
    CropForegroundd(
        keys=["image", "label"],
        source_key="image",
        select_fn=lambda x: x > 0,  # Or your specific threshold
        margin=0
    ),

    # crop center
    SpatialPadd(keys=["image"] + SEGMENTATION, spatial_size=(128, 128, -1,),
                allow_missing_keys=True),
    CenterSpatialCropd(keys=["image"] + SEGMENTATION, roi_size=(128, 128, -1,),
                       allow_missing_keys=True),

    # remove extreme values
    ClipIntensityPercentilesd(keys=["image"], lower=1, upper=99, channel_wise=True),

    # ensure right format
    CastToTyped(keys=["image", "label"], dtype=(torch.int16, torch.uint8))
])


def preprocess():
    for sample in tqdm(glob.glob('./data/BraTS/V2024/training_data1_v2/*'), desc="preprocessing"):
        print(sample)
        name = Path(sample).name

        volume_dir = f"./data/BraTS/preprocessed/{name}"
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

        mean_vals = torch.mean(image, dim=(1, 2, 3)).numpy()
        std_vals = torch.std(image, dim=(1, 2, 3)).numpy()

        for s in range(image.shape[-1]):
            slice_image = image[..., s].numpy()
            slice_label = label[..., s].numpy()

            save_path = os.path.join(volume_dir, f"slice{s:03d}.npz")

            np.savez_compressed(
                save_path,
                image=slice_image,
                label=slice_label,
                mean=mean_vals,
                std=std_vals
            )


def main():
    print(f"current working directory: {os.getcwd()}")

    syn = synapseclient.Synapse()
    syn.login()

    filepath = Path('./data/BraTS/BraTS2024-BraTS-GLI-TrainingData.zip')
    if not filepath.is_file():
        get(
            syn,
            "syn60086071",
            file_options=FileOptions(download_location='./data/BraTS')
        )

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

    print(f"{image.shape=}")
    print(f"{label.shape=}")

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