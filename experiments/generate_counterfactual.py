import glob

import hydra
import torch
from monai.networks.nets import BasicUNet
from omegaconf import DictConfig, OmegaConf

from datasets.base import get_datasets
from diff_cf.core.baseline.generator import BaselineGenerator
from diff_cf.core.diffusion.engine import Engine
from diff_cf.core.diffusion.generator import CounterfactualGenerator
from diff_cf.models.diffusion_model_final import DiffusionModel
from experiments.utils import visualize_mri, visualize_rgb


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    log_dir = f'{cfg.system.log_dir}/Generation'
    if cfg.generation.name is not None:
        run_name = cfg.generation.name
    else:
        nr = len(glob.glob(f"{log_dir}/*"))
        run_name = f"{nr}_counterfactual"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    _, dataset = get_datasets(cfg)
    print(f"dataset size: {len(dataset)}")

    if cfg.generation.sample_name:
        n = dataset.find_index(str(cfg.generation.sample_name))
    elif cfg.generation.sample_idx:
        n = cfg.generation.sample_idx
    else:
        raise ValueError("Either 'sample_idx' or 'sample_name' must be provided.")

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
            robust_model = BasicUNet(
                spatial_dims=2,  # Set to 2 for 2D image data (defaults to 3)
                in_channels=cfg.data.image_channels,
                out_channels=cfg.data.n_classes,
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
                                            log=True,
                                            name=run_name)

    else:
        generator = BaselineGenerator(segmentation, device, cfg, log=True, name=run_name)

    counterfactual, adaption, prediction_org, prediction_cf = generator.generate(image, target)

    if cfg.data.image_type == 'mri':
        visualize_mri(dataset, n, run_name, counterfactual, adaption, prediction_org, prediction_cf,
                      cfg)
    else:
        visualize_rgb(dataset, n, run_name, counterfactual, adaption, prediction_org, prediction_cf,
                      cfg)


if __name__ == '__main__':
    main()
