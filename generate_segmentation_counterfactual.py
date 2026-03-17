import os

import hydra
import torch

from omegaconf import DictConfig
import torchvision

from core.CounterfactualGenerator import CounterfactualGenerator
from core.DiffusionEngine import DiffusionEngine
from core.SegmentationDataset import SegmentationDataset
from models.DiffusionModel import DiffusionModel
from models.Segmentation import SegmentationUNet
from utils import visualize_mri, visualize_seg


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig, n :int = 685):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    dataset = SegmentationDataset()
    print(f"dataset size: {len(dataset)}")

    image = torch.stack([dataset[n][0]], dim=0)
    target = torch.stack([dataset[n][1]], dim=0)

    model = SegmentationUNet(
        in_channels=cfg.data.image_channels,
        out_channels=cfg.data.n_classes,
        img_width=cfg.data.image_width,
        img_height=cfg.data.image_height,
    )
    model.load_state_dict(torch.load(cfg.inference.segmentation_path, map_location=device))

    diffusion = DiffusionModel(cfg.model.dims, cfg.data.image_width, cfg.data.image_height,
                               cfg.model.time_dim)
    checkpoint = torch.load(cfg.inference.diffusion_path, map_location=device)
    diffusion.load_state_dict(checkpoint['model_state_dict'])

    engine = DiffusionEngine(**cfg.engine, device=device)

    generator = CounterfactualGenerator(model, diffusion, engine, device,
                                        weight_classifier=cfg.inference.weight_classifier,
                                        weight_distance=cfg.inference.weight_distance)

    counterfactual = generator.generate_counterfactual(image, target, cfg.inference.start_fraction)

    pred_original = model(image).detach().cpu().argmax(dim=1)
    pred_counterfactual = model(counterfactual).detach().cpu().argmax(dim=1)

    target = visualize_seg(target)
    pred_original = visualize_seg(pred_original)
    pred_counterfactual = visualize_seg(pred_counterfactual)

    image = visualize_mri(image)
    counterfactual = visualize_mri(counterfactual)

    os.makedirs(f"figures/example_{n}", exist_ok=True)

    torchvision.utils.save_image(target, f"figures/example_{n}/target.png")
    torchvision.utils.save_image(pred_original, f"figures/example_{n}/prediction_original.png")
    torchvision.utils.save_image(pred_counterfactual, f"figures/example_{n}/prediction_counterfactual.png")

    for i, channel in enumerate(["T1", "T1ce", "T2", "FLAIR"]):
        torchvision.utils.save_image(image[0, i], f"figures/example_{n}/original_{channel}.png")
        torchvision.utils.save_image(counterfactual[0, i], f"figures/example_{n}/counterfactual_{channel}.png")


if __name__ == '__main__':
    main()
