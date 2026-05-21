import glob
import os

import hydra
import torch

from omegaconf import DictConfig, OmegaConf
import torchvision
from torch import Tensor
from torchvision.utils import make_grid

from src.diff_cf.core.baseline.generator import BaselineGenerator
from src.diff_cf.core.datasets.brats_dataset import BraTSDataset
from src.diff_cf.core.datasets.celebamask_dataset import CelebAMaskDataset
from src.diff_cf.core.diffusion.generator import CounterfactualGenerator
from src.diff_cf.core.datasets.dataset import get_datasets
from src.diff_cf.core.diffusion.engine import Engine
from src.diff_cf.models.diffusion_model_new import DiffusionModel
from src.diff_cf.models.segmentation import SegmentationUNet
from src.diff_cf.utils.utils import visualize_seg, visualize_counterfactuals, visualize_adaption


def visualize_rgb(dataset: CelebAMaskDataset,
                  n: int,
                  run_name: str,
                  counterfactual: Tensor,
                  adaption: Tensor,
                  prediction_org: Tensor,
                  prediction_cf: Tensor,
                  cfg: DictConfig):
    image, target = dataset[n][0], dataset[n][1]
    image = dataset.denormalize(image)
    counterfactual = dataset.denormalize(counterfactual)

    target = visualize_seg(target).permute(2, 0, 1)
    pred_original = visualize_seg(prediction_org[0]).permute(2, 0, 1)
    pred_counterfactual = visualize_seg(prediction_cf[0]).permute(2, 0, 1)

    x_min = ((cfg.data.min - torch.tensor(cfg.data.dataset.means)) / torch.tensor(
        cfg.data.dataset.stds))
    x_max = ((cfg.data.max - torch.tensor(cfg.data.dataset.means)) / torch.tensor(
        cfg.data.dataset.stds))

    maximas = torch.maximum(torch.abs(x_min), torch.abs(x_max))

    adaption_visual = visualize_adaption(adaption, maximas)

    os.makedirs(f"{cfg.system.figures_dir}/{run_name}", exist_ok=True)

    with open(f"{cfg.system.figures_dir}/{run_name}/config.txt", "a") as f:
        f.write(str(cfg))

    torchvision.utils.save_image(target,
                                 f"{cfg.system.figures_dir}/{run_name}/target.png")
    torchvision.utils.save_image(pred_original,
                                 f"{cfg.system.figures_dir}/{run_name}/prediction_original.png")
    torchvision.utils.save_image(pred_counterfactual,
                                 f"{cfg.system.figures_dir}/{run_name}/prediction_counterfactual.png")

    torchvision.utils.save_image(image,
                                 f"{cfg.system.figures_dir}/{run_name}/original.png")
    torchvision.utils.save_image(counterfactual[0],
                                 f"{cfg.system.figures_dir}/{run_name}/counterfactual.png")

    for i, channel in enumerate(["red", "green", "blue"]):
        torchvision.utils.save_image(adaption_visual[0, i],
                                     f"{cfg.system.figures_dir}/{run_name}/adaption_{channel}.png")

    adaption_visual = visualize_adaption(adaption.sum(dim=1, keepdim=True), maximas=torch.max(maximas))

    stacked_image = make_grid([
        image,
        adaption_visual[0, 0],
        counterfactual[0],
        pred_original,
        pred_counterfactual,
        target
    ], nrow=3)
    torchvision.utils.save_image(stacked_image,
                                 f"{cfg.system.figures_dir}/{run_name}/comparison.png")


def visualize_mri(dataset: BraTSDataset,
                  n: int,
                  run_name: str,
                  counterfactual: Tensor,
                  adaption: Tensor,
                  prediction_org: Tensor,
                  prediction_cf: Tensor,
                  cfg: DictConfig):
    image, target = dataset[n][0], dataset[n][1]

    target = visualize_seg(target)[0].permute(2, 0, 1)
    pred_original = visualize_seg(prediction_org)[0].permute(2, 0, 1)
    pred_counterfactual = visualize_seg(prediction_cf)[0].permute(2, 0, 1)

    adaption = visualize_adaption(adaption)
    image, counterfactual = visualize_counterfactuals(image, counterfactual)

    os.makedirs(f"{cfg.system.figures_dir}/{run_name}", exist_ok=True)

    with open(f"{cfg.system.figures_dir}/{run_name}/config.txt", "a") as f:
        f.write(str(cfg))

    torchvision.utils.save_image(target, f"{cfg.system.figures_dir}/{run_name}/target.png")
    torchvision.utils.save_image(pred_original,
                                 f"{cfg.system.figures_dir}/{run_name}/prediction_original.png")
    torchvision.utils.save_image(pred_counterfactual,
                                 f"{cfg.system.figures_dir}/{run_name}/prediction_counterfactual.png")

    for i, channel in enumerate(["T1", "T1ce", "T2", "FLAIR"]):
        torchvision.utils.save_image(image[0, i],
                                     f"{cfg.system.figures_dir}/{run_name}/original_{channel}.png")
        torchvision.utils.save_image(counterfactual[0, i],
                                     f"{cfg.system.figures_dir}/{run_name}/counterfactual_{channel}.png")
        torchvision.utils.save_image(adaption[0, i],
                                     f"{cfg.system.figures_dir}/{run_name}/adaption_{channel}.png")

    stacked_image = make_grid(torch.stack([
        image[0],
        adaption[0],
        counterfactual[0]
    ]).flatten(0, 1), nrow=4)
    torchvision.utils.save_image(stacked_image,
                                 f"{cfg.system.figures_dir}/{run_name}/comparison.png")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    log_dir = f'{cfg.system.log_dir}/Generation'
    if cfg.generation.name is not None:
        run_name = cfg.generation.name
    else:
        nr = len(glob.glob(f"{log_dir}/*"))
        run_name = f"{nr}_counterfactual"

    n = cfg.generation.sample_idx

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    _, dataset = get_datasets(cfg)
    print(f"dataset size: {len(dataset)}")

    image = torch.stack([dataset[n][0]], dim=0)
    target = torch.stack([dataset[n][1]], dim=0)

    segmentation = SegmentationUNet(
        in_channels=cfg.data.image_channels,
        out_channels=cfg.data.n_classes,
        img_width=cfg.data.image_width,
        img_height=cfg.data.image_height,
        attention=cfg.segmentation.attention,
    )
    checkpoint = torch.load(
        f"{cfg.segmentation.checkpoint_path}/{cfg.segmentation.name}.pt",
        map_location=device
    )
    segmentation.load_state_dict(checkpoint['model_state_dict'])

    if cfg.generation.type == 'Diffusion':
        diffusion = DiffusionModel(cfg.data.image_channels, cfg.data.image_width, cfg.data.image_height,
                                   cfg.diffusion.time_dim)

        checkpoint = torch.load(
            f"{cfg.diffusion.checkpoint_path}/{cfg.diffusion.name}.pt",
            map_location=device
        )
        diffusion.load_state_dict(checkpoint['model_state_dict'])

        engine = Engine(cfg, device=device)

        generator = CounterfactualGenerator(segmentation, diffusion, engine, device, cfg, name=run_name)

    else:
        generator = BaselineGenerator(segmentation, device, cfg, name=run_name)


    counterfactual, adaption, prediction_org, prediction_cf = generator.generate(image, target)

    if cfg.data.image_type == 'mri':
        visualize_mri(dataset, n, run_name, counterfactual, adaption, prediction_org, prediction_cf, cfg)
    else:
        visualize_rgb(dataset, n, run_name, counterfactual, adaption, prediction_org, prediction_cf, cfg)


if __name__ == '__main__':
    main()
