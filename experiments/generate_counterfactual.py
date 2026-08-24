import glob
import os
import hydra
import torch
from monai.networks.nets import BasicUNet

from omegaconf import DictConfig, OmegaConf
import torchvision
from torch import Tensor
from torchvision.utils import make_grid

from diff_cf.core.baseline.generator import BaselineGenerator
from datasets.brats import BraTSDataset
from datasets.celebamask import CelebAMaskDataset
from diff_cf.core.diffusion.generator import CounterfactualGenerator
from datasets.base import get_datasets
from diff_cf.core.diffusion.engine import Engine
from diff_cf.models.diffusion_model_final import DiffusionModel
from diff_cf.models.segmentation import SegmentationUNet
from diff_cf.utils.utils import visualize_seg, visualize_counterfactuals, visualize_adaption


def visualize_rgb(dataset: CelebAMaskDataset,
                  n: int,
                  run_name: str,
                  counterfactual: Tensor,
                  adaption: Tensor,
                  prediction_org: Tensor,
                  prediction_cf: Tensor,
                  cfg: DictConfig):
    image, target = dataset[n][0], dataset[n][1]

    print("Before denormalize:")
    print("Counterfactual:")
    print("min: ", counterfactual.min(), "max: ", counterfactual.max())
    print((counterfactual < 0).float().mean(), (counterfactual > 1).float().mean())

    print("Image:")
    print("min: ", image.min(), "max: ", image.max())
    print((image < 0).float().mean(), (image > 1).float().mean())

    image = dataset.denormalize(image) / 255.0
    counterfactual = dataset.denormalize(counterfactual[0]) / 255.0

    print("\nAfter denormalize:")
    print("Counterfactual:")
    print("min: ", counterfactual.min(), "max: ", counterfactual.max())
    print((counterfactual < 0).float().mean(), (counterfactual > 1).float().mean())

    print("\n Image:")
    print("min: ", image.min(), "max: ", image.max())
    print((image < 0).float().mean(), (image > 1).float().mean())

    target = visualize_seg(target)
    pred_original = visualize_seg(prediction_org[0])
    pred_counterfactual = visualize_seg(prediction_cf[0])

    x_min = ((cfg.data.min - torch.tensor(cfg.data.means)) / torch.tensor(
        cfg.data.stds))
    x_max = ((cfg.data.max - torch.tensor(cfg.data.means)) / torch.tensor(
        cfg.data.stds))

    maximas = torch.maximum(torch.abs(x_min), torch.abs(x_max))

    adaption_visual = visualize_adaption(adaption, maximas)

    os.makedirs(f"{cfg.system.figures_dir}/{run_name}", exist_ok=True)

    with open(f"{cfg.system.figures_dir}/{run_name}/config.txt", "a") as f:
        f.write(OmegaConf.to_yaml(cfg))

    torchvision.utils.save_image(target,
                                 f"{cfg.system.figures_dir}/{run_name}/target.png")
    torchvision.utils.save_image(pred_original,
                                 f"{cfg.system.figures_dir}/{run_name}/prediction_original.png")
    torchvision.utils.save_image(pred_counterfactual,
                                 f"{cfg.system.figures_dir}/{run_name}/prediction_counterfactual.png")

    torchvision.utils.save_image(image,
                                 f"{cfg.system.figures_dir}/{run_name}/original.png")
    torchvision.utils.save_image(counterfactual,
                                 f"{cfg.system.figures_dir}/{run_name}/counterfactual.png")

    for i, channel in enumerate(["red", "green", "blue"]):
        torchvision.utils.save_image(adaption_visual[0, i],
                                     f"{cfg.system.figures_dir}/{run_name}/adaption_{channel}.png")

    adaption_visual = visualize_adaption(adaption.sum(dim=1, keepdim=True),
                                         maximas=torch.max(maximas))

    stacked_image = make_grid([
        image,
        adaption_visual[0, 0],
        counterfactual,
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

    target = visualize_seg(target)[0]
    pred_original = visualize_seg(prediction_org)[0]
    pred_counterfactual = visualize_seg(prediction_cf)[0]

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

    segmentation = BasicUNet(
        spatial_dims=2,  # Set to 2 for 2D image data (defaults to 3)
        in_channels=cfg.data.image_channels,
        out_channels=cfg.data.n_classes,
    )

    checkpoint = torch.load(
        f"{cfg.segmentation.checkpoint_path}/{cfg.generation.model}.pt",
        map_location=device
    )
    segmentation.load_state_dict(checkpoint['model_state_dict'])

    if cfg.generation.type == 'Diffusion':
        if cfg.generation.robust is not None:
            robust_model = SegmentationUNet(
                in_channels=cfg.data.image_channels,
                out_channels=cfg.data.n_classes,
                img_width=cfg.data.image_width,
                img_height=cfg.data.image_height,
                attention=cfg.segmentation.attention,
            )
            checkpoint = torch.load(
                f"{cfg.segmentation.checkpoint_path}/{cfg.generation.robust}.pt",
                map_location=device
            )
            robust_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            robust_model = None

        diffusion = DiffusionModel(channels=cfg.data.image_channels,
                                   width=cfg.data.image_width,
                                   height=cfg.data.image_height,
                                   time_dim=cfg.diffusion.time_dim,
                                   nums_features=cfg.diffusion.nums_features,
                                   use_prior=cfg.diffusion.use_target,
                                   num_classes=cfg.data.n_classes)

        checkpoint = torch.load(
            f"{cfg.diffusion.checkpoint_path}/{cfg.diffusion.name}.pt",
            map_location=device
        )
        diffusion.load_state_dict(checkpoint['model_state_dict'])

        engine = Engine(cfg, device=device)

        generator = CounterfactualGenerator(segmentation,
                                            diffusion,
                                            engine,
                                            device,
                                            cfg,
                                            robust_model,
                                            name=run_name)

    else:
        generator = BaselineGenerator(segmentation, device, cfg, name=run_name)

    counterfactual, adaption, prediction_org, prediction_cf = generator.generate(image, target)

    if cfg.data.image_type == 'mri':
        visualize_mri(dataset, n, run_name, counterfactual, adaption, prediction_org, prediction_cf,
                      cfg)
    else:
        visualize_rgb(dataset, n, run_name, counterfactual, adaption, prediction_org, prediction_cf,
                      cfg)


if __name__ == '__main__':
    main()
