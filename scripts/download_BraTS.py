import os

import hydra
import numpy as np
import torch
from monai.apps import DecathlonDataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, CastToTyped, \
    NormalizeIntensityd, SpatialPadd, CenterSpatialCropd, ClipIntensityPercentilesd
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(version_base=None, config_path="../conf/data", config_name="BraTS")
def main(cfg: DictConfig):
    root_dir = cfg.root
    output_dir = os.path.join(root_dir, "Slices")
    os.makedirs(output_dir, exist_ok=True)

    KEYS = ["image", "label"]

    # Define the transforms
    transforms = Compose([
        LoadImaged(keys=KEYS, allow_missing_keys=True),
        EnsureChannelFirstd(keys=KEYS, allow_missing_keys=True),

        # crop center
        SpatialPadd(keys=KEYS, spatial_size=(cfg.image_width, cfg.image_height, -1,), allow_missing_keys=True),
        CenterSpatialCropd(keys=KEYS, roi_size=(cfg.image_width, cfg.image_height, -1,), allow_missing_keys=True),

        # scale intensities
        ClipIntensityPercentilesd(
            keys=["image"],
            lower=1, upper=99),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

        # ensure right format
        CastToTyped(keys=KEYS, dtype=(torch.float32, torch.uint8))
    ])

    dataset = DecathlonDataset(
        root_dir=root_dir,
        task="Task01_BrainTumour",
        section="training",
        transform=transforms,
        download=True,
        cache_num=10,
        num_workers=4
    )

    for sample in tqdm(dataset, desc=f"Processing"):
        image = sample["image"].numpy()
        num_slices = image.shape[-1]
        base_name = os.path.basename(sample["image"].meta["filename_or_obj"]).split('.')[-3]

        if "label" in sample.keys():
            label = sample["label"].numpy()
            has_mask_flag = True
        else:
            label = np.zeros((1, *image.shape[1:]), dtype=np.uint8)
            has_mask_flag = False

        volume_dir = os.path.join(output_dir, "training", base_name)
        os.makedirs(volume_dir, exist_ok=True)

        for s in range(num_slices):
            slice_image = image[..., s]
            slice_label = label[..., s]

            save_path = os.path.join(volume_dir, f"slice{s:03d}.npz")

            np.savez(
                save_path,
                image=slice_image,
                label=slice_label,
                has_mask=has_mask_flag
            )


if __name__ == '__main__':
    main()
