import os
from pathlib import Path

import hydra
import pandas as pd
import torch
from monai.networks.nets import BasicUNet
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image
from tqdm import tqdm

from datasets.base import get_datasets
from datasets.celebamask import get_ood_datasets
from diff_cf.core.segmentation_trainer import SegmentationTrainer
from diff_cf.utils.utils import visualize_seg
from diff_cf.utils.metrics import calc_iou, calc_f1_score

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Setup Logging & Run Number
    log_dir = f'{cfg.system.log_dir}/Segmentation'
    run_name = cfg.segmentation.name
    print(f"{run_name=}")

    # check for gpu available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    train_dataset, test_dataset = get_datasets(cfg, cfg.segmentation.augmentations.enable)
    if cfg.segmentation.ood.enable:
        train_dataset.set_attribute(cfg.segmentation.ood.attribute,
                                    use_positive=cfg.segmentation.ood.train_on_positive)
        test_dataset.set_attribute(cfg.segmentation.ood.attribute,
                                   use_positive=~cfg.segmentation.ood.train_on_positive)

    print(f"{len(train_dataset)} train samples")
    print(f"{len(test_dataset)} test samples")

    print(train_dataset[0][0].shape)

    model = BasicUNet(
        spatial_dims=2,  # Set to 2 for 2D image data (defaults to 3)
        in_channels=cfg.data.image_channels,
        out_channels=cfg.data.n_classes,
    ).to(device)

    trainer = SegmentationTrainer(
        train_dataset,
        test_dataset,
        model,
        device,
        cfg,
        name=run_name,
        log_dir=log_dir
    )

    checkpoint = f'{cfg.segmentation.checkpoint_path}/{run_name}.pt'
    if Path(checkpoint).is_file():
        trainer.load_checkpoint(checkpoint)
        print(f"loaded model form {checkpoint}...")

    results = []
    for i, (image, target) in tqdm(enumerate(test_dataset), total=len(test_dataset), desc="segmentation"):
        pred = trainer.predict(image[None].to(device))
        iou = calc_iou(target, pred, num_classes=cfg.data.n_classes)
        f1, precision, recall = calc_f1_score(target, pred, num_classes=cfg.data.n_classes)

        file = Path(test_dataset.get_filepath(i)).stem
        path = f"{cfg.system.figures_dir}/segmentation/{file}"
        os.makedirs(f"{cfg.system.figures_dir}/segmentation/{file}", exist_ok=True)

        save_image(visualize_seg(pred[0]), f"{path}/prediction.png")
        save_image(visualize_seg(target), f"{path}/target.png")
        save_image(test_dataset.denormalize(image, i) / 256, f"{path}/image.png")
        results.append({'file': f"{file}",
                        'iou': iou,
                        'f1': f1,
                        'precision': precision,
                        'recall': recall})

    pd.DataFrame(results, columns=['path', 'iou', 'f1', 'precision', 'recall']).to_csv(
        f"{cfg.system.figures_dir}/segmentation/results.csv", index=False)


if __name__ == '__main__':
    main()
