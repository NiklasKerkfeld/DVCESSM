import glob
import time

from hydra.utils import instantiate

import hydra
import torch
from monai.networks.nets import BasicUNet
from omegaconf import DictConfig, OmegaConf

from diff_cf.core.base_dataset import get_splits
from diff_cf.core.baseline.generator import BaselineGenerator
from diff_cf.core.diffusion.generator import CounterfactualGenerator
from diff_cf.models.diffusion_model import DiffusionModel
from experiments.utils import visualize_mri, visualize_rgb


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # print config
    print(OmegaConf.to_yaml(cfg))

    # set random seeds:
    if cfg.generation.seed is not None:
        torch.random.manual_seed(cfg.generation.seed)

    # setup log_dir and run_name
    if cfg.generation.name is not None:
        run_name = str(cfg.generation.name)
    else:
        nr = len(glob.glob(f"{cfg.system.log_dir}/*"))
        run_name = f"{nr}_counterfactual"

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # load test dataset
    _, test_data = get_splits(root=cfg.data.root)
    dataset = instantiate(cfg.data.dataset, samples=test_data)
    print(f"dataset size: {len(dataset)}")


    # get sample by name or by index
    if cfg.generation.sample_name:
        n = dataset.find_index(str(cfg.generation.sample_name))
        if n is None:
            raise FileNotFoundError(f"File {str(cfg.generation.sample_name)} not found in dataset!")
    elif cfg.generation.sample_idx:
        n = cfg.generation.sample_idx
    else:
        raise ValueError("Either 'sample_idx' or 'sample_name' must be provided.")

    image = torch.stack([dataset[n][0]], dim=0)
    target = torch.stack([dataset[n][1]], dim=0)

    # init segmentation model and load weights
    segmentation_model = BasicUNet(
        spatial_dims=2,  # Set to 2 for 2D image data (defaults to 3)
        in_channels=cfg.data.image_channels,
        out_channels=cfg.data.n_classes,
    )

    checkpoint = torch.load(
        f"{cfg.segmentation.checkpoint_path}/{cfg.generation.model}.pt",
        map_location=device
    )
    segmentation_model.load_state_dict(checkpoint['model_state_dict'])

    # Losses
    segmentation_loss_config = instantiate(cfg.generation.SegmentationLoss)
    distance_loss = instantiate(cfg.generation.distance_loss)

    # AdaptGrad
    adapt_grad = instantiate(cfg.generation.AdaptGrad, z_min=dataset.z_min, z_max=dataset.z_max)

    if cfg.generation.type == 'Diffusion':
        # Smoothing
        smoothing_cfg = instantiate(cfg.generation.GradSmoothing)

        # Init robust model if given
        robust_model = None
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

        # init diffusion model and load weights
        diffusion_model = DiffusionModel(channels=cfg.data.image_channels,
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
        diffusion_model.load_state_dict(checkpoint['model_state_dict'])

        # init diffusion engine
        engine = instantiate(cfg.engine, device=device)

        # init diffusion counterfactual generator
        generator = CounterfactualGenerator(
            segmentation_model=segmentation_model,
            diffusion_model=diffusion_model,
            engine=engine,
            segmentation_loss_config=segmentation_loss_config,
            distance_loss=distance_loss,
            z_bounds=(dataset.z_min, dataset.z_max),
            use_target_prior=cfg.diffusion.use_target,
            robust_model=robust_model,
            weight_segmentation=cfg.generation.weight_segmentation,
            weight_distance=cfg.generation.weight_distance,
            start_fraction=cfg.generation.start_fraction,
            approximation_step_size=cfg.generation.approximation_step_size,
            adapt_grad=adapt_grad,
            smoothing_cfg=smoothing_cfg,
            n_classes=cfg.data.n_classes,
            cone_projection_alpha=cfg.generation.cone_projection_alpha,
            log_dir=cfg.system.log_dir,
            log_to_tensorboard=True,
            device=device,
            name=run_name,
        )

    else:
        # Alternative load Baseline Generator
        generator = BaselineGenerator(
            segmentation_model=segmentation_model,
            segmentation_loss_config=segmentation_loss_config,
            distance_loss=distance_loss,
            z_bounds=(dataset.z_min, dataset.z_max),
            adapt_grad=adapt_grad,
            lr=cfg.generation.lr,
            steps=cfg.generation.steps,
            alpha=cfg.generation.alpha,
            beta=cfg.generation.beta,
            cone_projection_alpha=cfg.generation.cone_projection_alpha,
            n_classes=cfg.data.n_classes,
            log_dir=cfg.system.log_dir,
            log_to_tensorboard=True,
            device=device,
            name=run_name,
        )

    start_time = time.time()
    counterfactual, adaption, prediction_org, prediction_cf = generator.generate(image, target)
    time_needed = time.time() - start_time

    if cfg.data.image_type == 'mri':
        visualize_mri(dataset, n, run_name, counterfactual, adaption, prediction_org, prediction_cf,
                      cfg)
    else:
        visualize_rgb(dataset, n, run_name, counterfactual, adaption, prediction_org, prediction_cf,
                      time_needed, cfg)


if __name__ == '__main__':
    main()
