import os

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from core.Datasets import get_datasets
from core.GAN.Generator import CounterfactualGenerator
from models.GAN import Generator, Adversarial
from models.Segmentation import SegmentationUNet
from utils import calculate_iou, visualize_seg, visualize_mri, visualize_adaption

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Setup Logging & Run Number
    if cfg.gan.name is not None:
        run_name = cfg.gan.name
    else:
        raise ValueError(f"Parameter gan.name not defined!")

    # check for gpu available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    _, test_dataset = get_datasets('Segmentation')

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,  # Speeds up transfer to GPU
        drop_last=True
    )

    generator = CounterfactualGenerator(
        generator=Generator(cfg.data.image_channels, cfg.data.n_classes),
        adversarial=Adversarial(cfg.data.image_channels, cfg.data.n_classes, cfg.data.image_width,
                                cfg.data.image_height),
        segmentation=SegmentationUNet(
            in_channels=cfg.data.image_channels,
            out_channels=cfg.data.n_classes,
            img_width=cfg.data.image_width,
            img_height=cfg.data.image_height,
            attention=cfg.segmentation.attention,
        ),
        checkpoint=f"{cfg.gan.checkpoint_path}/{run_name}.pt",
        device=device
    )

    classes = ['ET', 'SNFH', 'NETC', 'RC']

    bar = tqdm(test_loader, total=len(test_loader))
    count = 0
    for i, (image, target) in enumerate(bar):
        image.to(device)
        target.to(device)
        if target.sum() == 0.0:
            continue

        counterfactual, adaption, prediction_org, prediction_cf = generator.generate(
            image, target)

        cls_iou_org, mean_iou_org = calculate_iou(prediction_org, target,
                                                  num_classes=cfg.data.n_classes,
                                                  ignore_index=0)

        cls_iou_cf, mean_iou_cf = calculate_iou(prediction_cf, target,
                                                num_classes=cfg.data.n_classes,
                                                ignore_index=0)

        image = visualize_mri(image)
        counterfactual = visualize_mri(counterfactual)
        adaption = visualize_adaption(adaption)
        prediction_org = visualize_seg(prediction_org)
        prediction_cf = visualize_seg(prediction_cf)
        target = visualize_seg(target)

        os.makedirs(f"./figures/evaluation/image_{i}", exist_ok=True)
        save_image(image[0], f"./figures/evaluation/image_{i}/image.png")
        save_image(prediction_org[0].permute(2, 0, 1),
                   f"./figures/evaluation/image_{i}/prediction_org.png")
        save_image(adaption[0],
                   f"./figures/evaluation/image_{i}/adaption.png")
        save_image(counterfactual[0], f"./figures/evaluation/image_{i}/counterfactual.png")
        save_image(prediction_cf[0].permute(2, 0, 1),
                   f"./figures/evaluation/image_{i}/prediction_cf.png")
        save_image(target[0].permute(2, 0, 1), f"./figures/evaluation/image_{i}/target.png")

        with open(f"./figures/evaluation/image_{i}/results.txt", "w") as f:
            # Writing individual class IoUs
            f.write("Original Segmentation:\n")
            for i, iou_val in enumerate(cls_iou_org):
                f.write(f"\tClass {classes[i]}: {iou_val:.4f}\n")
            f.write(f"\tMean IoU: {mean_iou_org:.4f}\n")

            f.write("Counterfactual Segmentation:\n")
            for i, iou_val in enumerate(cls_iou_cf):
                f.write(f"\tClass {classes[i]}: {iou_val:.4f}\n")
            f.write(f"\tMean IoU: {mean_iou_cf:.4f}\n")

        bar.set_description(f"{count} samples")


if __name__ == '__main__':
    main()
