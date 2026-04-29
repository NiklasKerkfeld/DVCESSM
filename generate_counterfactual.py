import os

import hydra
import torch

from omegaconf import DictConfig
import torchvision
from torchvision.utils import make_grid

from core.Diffusion.Generator import CounterfactualGenerator
from core.Datasets.dataset import get_datasets
from core.Diffusion.Engine import Engine
from models.DiffusionModel_new import DiffusionModel
from models.Segmentation import SegmentationUNet
from utils import visualize_seg, visualize_counterfactuals, visualize_adaption


def visualize_rgb(dataset, n, counterfactual, adaption, prediction_org, prediction_cf, cfg):
    image, target = dataset[n][0], dataset[n][1]
    image = dataset.denormalize(image)

    target = visualize_seg(target).permute(2, 0, 1)
    pred_original = visualize_seg(prediction_org[0]).permute(2, 0, 1)
    pred_counterfactual = visualize_seg(prediction_cf[0]).permute(2, 0, 1)

    adaption_visual = visualize_adaption(adaption)

    print(f"{target.shape=}, {pred_original.shape=}, {pred_counterfactual.shape=}")

    os.makedirs(f"{cfg.system.figures_dir}/example_{n}", exist_ok=True)

    torchvision.utils.save_image(target,
                                 f"{cfg.system.figures_dir}/example_{n}/target.png")
    torchvision.utils.save_image(pred_original,
                                 f"{cfg.system.figures_dir}/example_{n}/prediction_original.png")
    torchvision.utils.save_image(pred_counterfactual,
                                 f"{cfg.system.figures_dir}/example_{n}/prediction_counterfactual.png")

    torchvision.utils.save_image(image,
                                 f"{cfg.system.figures_dir}/example_{n}/original.png")
    torchvision.utils.save_image(counterfactual[0],
                                 f"{cfg.system.figures_dir}/example_{n}/counterfactual.png")

    for i, channel in enumerate(["red", "green", "blue"]):
        torchvision.utils.save_image(adaption_visual[0, i],
                                     f"{cfg.system.figures_dir}/example_{n}/adaption_{channel}.png")

    adaption_visual = visualize_adaption(adaption.sum(dim=1, keepdim=True))

    print(f"{image.shape=}, {adaption_visual.shape=}, {counterfactual.shape=}")
    stacked_image = make_grid([
        image,
        adaption_visual[0, 0],
        counterfactual[0],
        pred_original,
        pred_counterfactual,
        target
    ], nrow=3)
    print(f"{stacked_image.shape=}")
    torchvision.utils.save_image(stacked_image,
                                 f"{cfg.system.figures_dir}/example_{n}/comparison.png")


def visualize_mri(dataset, n, counterfactual, adaption, prediction_org, prediction_cf, cfg):
    image, target = dataset[n][0], dataset[n][1]

    target = visualize_seg(target)[0].permute(2, 0, 1)
    pred_original = visualize_seg(prediction_org)[0].permute(2, 0, 1)
    pred_counterfactual = visualize_seg(prediction_cf)[0].permute(2, 0, 1)

    adaption = visualize_adaption(adaption)
    image, counterfactual = visualize_counterfactuals(image, counterfactual)

    os.makedirs(f"{cfg.system.figures_dir}/example_{n}", exist_ok=True)
    torchvision.utils.save_image(target, f"{cfg.system.figures_dir}/example_{n}/target.png")
    torchvision.utils.save_image(pred_original,
                                 f"{cfg.system.figures_dir}/example_{n}/prediction_original.png")
    torchvision.utils.save_image(pred_counterfactual,
                                 f"{cfg.system.figures_dir}/example_{n}/prediction_counterfactual.png")

    for i, channel in enumerate(["T1", "T1ce", "T2", "FLAIR"]):
        torchvision.utils.save_image(image[0, i],
                                     f"{cfg.system.figures_dir}/example_{n}/original_{channel}.png")
        torchvision.utils.save_image(counterfactual[0, i],
                                     f"{cfg.system.figures_dir}/example_{n}/counterfactual_{channel}.png")
        torchvision.utils.save_image(adaption[0, i],
                                     f"{cfg.system.figures_dir}/example_{n}/adaption_{channel}.png")

    stacked_image = make_grid(torch.stack([
        image[0],
        adaption[0],
        counterfactual[0]
    ]).flatten(0, 1), nrow=4)
    torchvision.utils.save_image(stacked_image,
                                 f"{cfg.system.figures_dir}/example_{n}/comparison.png")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    n = cfg.diffusion.sample_idx

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    _, dataset = get_datasets(cfg)
    print(f"dataset size: {len(dataset)}")

    image = torch.stack([dataset[n][0]], dim=0).to(device)
    target = torch.stack([dataset[n][1]], dim=0).to(device)

    segmentation = SegmentationUNet(
        in_channels=cfg.data.image_channels,
        out_channels=cfg.data.n_classes,
        img_width=cfg.data.image_width,
        img_height=cfg.data.image_height,
        attention=cfg.segmentation.attention,
    ).to(device)
    diffusion = DiffusionModel(cfg.data.image_channels, cfg.data.image_width, cfg.data.image_height,
                               cfg.diffusion.time_dim).to(device)

    engine = Engine(cfg, device=device).to(device)

    generator = CounterfactualGenerator(segmentation, diffusion, engine, device, cfg)

    counterfactual, adaption, prediction_org, prediction_cf = generator.generate(image, target,
                                                                                 cfg.diffusion.start_fraction)

    if cfg.data.image_type == 'mri':
        visualize_mri(dataset, n, counterfactual, adaption, prediction_org, prediction_cf, cfg)
    else:
        visualize_rgb(dataset, n, counterfactual, adaption, prediction_org, prediction_cf, cfg)


if __name__ == '__main__':
    main()
