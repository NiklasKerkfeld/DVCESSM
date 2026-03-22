import os

import hydra
import torch

from omegaconf import DictConfig
import torchvision

from core.Diffusion.Generator import CounterfactualGenerator
from core.Datasets import get_datasets
from core.Diffusion.Engine import Engine
from models.DiffusionModel import DiffusionModel
from models.Segmentation import SegmentationUNet
from utils import visualize_mri, visualize_seg


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig, n :int = 585):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    _, dataset = get_datasets('Segmentation')
    print(f"dataset size: {len(dataset)}")

    image = torch.stack([dataset[n][0]], dim=0)
    target = torch.stack([dataset[n][1]], dim=0)

    segmentation = SegmentationUNet(
        in_channels=cfg.data.image_channels,
        out_channels=cfg.data.n_classes,
        img_width=cfg.data.image_width,
        img_height=cfg.data.image_height,
    )
    diffusion = DiffusionModel(cfg.model.dims, cfg.data.image_width, cfg.data.image_height,
                               cfg.model.time_dim)


    engine = Engine(**cfg.diffusion.engine, device=device)

    generator = CounterfactualGenerator(segmentation, diffusion, engine, device,
                                        weight_classifier=cfg.inference.weight_classifier,
                                        weight_distance=cfg.inference.weight_distance)

    generator.load_checkpoint(f"{cfg.diffusion.checkpoint_path}/{cfg.diffusion.name}.pt", f"{cfg.inference.checkpoint_path}/{cfg.inference.name}.pt")

    counterfactual, adaption, prediction_org, prediction_cf = generator.generate(image, target, cfg.inference.start_fraction)

    target = visualize_seg(target)
    pred_original = visualize_seg(prediction_org)
    pred_counterfactual = visualize_seg(prediction_cf)

    image = visualize_mri(image)
    counterfactual = visualize_mri(counterfactual)

    os.makedirs(f"{cfg.system.figures_dir}/example_{n}", exist_ok=True)

    torchvision.utils.save_image(target, f"{cfg.system.figures_dir}/example_{n}/target.png")
    torchvision.utils.save_image(pred_original, f"{cfg.system.figures_dir}/example_{n}/prediction_original.png")
    torchvision.utils.save_image(pred_counterfactual, f"{cfg.system.figures_dir}/example_{n}/prediction_counterfactual.png")

    for i, channel in enumerate(["T1", "T1ce", "T2", "FLAIR"]):
        torchvision.utils.save_image(image[0, i], f"{cfg.system.figures_dir}/example_{n}/original_{channel}.png")
        torchvision.utils.save_image(counterfactual[0, i], f"{cfg.system.figures_dir}/example_{n}/counterfactual_{channel}.png")


if __name__ == '__main__':
    main()
