import os

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from core.SegmentationDataset import SegmentationDataset
from models.GAN import Generator, Adversarial
from models.Segmentation import SegmentationUNet
from utils import calculate_iou, visualize_seg, visualize_mri, visualize_adaption

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # print the resolved config to your terminal for sanity
    print(OmegaConf.to_yaml(cfg))

    # check for gpu available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    full_dataset = SegmentationDataset()
    _, test_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2])

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,  # Speeds up transfer to GPU
        drop_last=True
    )

    generator = Generator(cfg.data.image_channels, cfg.data.n_classes, scale=cfg.gan.scale)
    generator.load_state_dict(torch.load(f"checkpoints/run5_generator.pth", map_location=device))
    generator.eval()

    adversarial = Adversarial(cfg.data.image_channels, cfg.data.n_classes, cfg.data.image_width,
                              cfg.data.image_height)
    adversarial.load_state_dict(
        torch.load(f"checkpoints/run5_adversarial.pth", map_location=device))
    adversarial.eval()

    segmentation = SegmentationUNet(
        in_channels=cfg.data.image_channels,
        out_channels=cfg.data.n_classes,
        img_width=cfg.data.image_width,
        img_height=cfg.data.image_height,
    )
    segmentation.load_state_dict(torch.load(cfg.inference.segmentation_path, map_location=device))
    segmentation.eval()

    classes = ['Background', 'ET', 'SNFH', 'NETC', 'RC']

    bar = tqdm(test_loader, total=len(test_loader))
    count = 0
    for i, (image, target) in enumerate(bar):
        image.to(device)
        target.to(device)

        with torch.no_grad():
            prediction_org = segmentation(image).argmax(1).long()

            cls_iou_org, mean_iou_org = calculate_iou(prediction_org, target,
                                                      num_classes=cfg.data.n_classes)
            if mean_iou_org < 0.5:
                os.makedirs(f"./figures/evaluation/image_{i}/", exist_ok=True)
                count += 1

                adaption = generator(image, target, prediction_org)
                counterfactual = image + adaption
                prediction_cf = segmentation(counterfactual).argmax(1).long()

                cls_iou_cf, mean_iou_cf = calculate_iou(prediction_cf, target,
                                                        num_classes=cfg.data.n_classes,
                                                        ignore_index=0)

                image = visualize_mri(image)
                counterfactual = visualize_mri(counterfactual)
                adaption = visualize_adaption(adaption)
                prediction_org = visualize_seg(prediction_org)
                prediction_cf = visualize_seg(prediction_cf)
                target = visualize_seg(target)

                save_image(image[0], f"./figures/evaluation/image_{i}/image.png")
                save_image(prediction_org[0].permute(2, 0, 1),
                           f"./figures/evaluation/image_{i}/prediction_org.png")
                save_image(adaption[0].permute(0, 3, 1, 2),
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

                bar.set_postfix(f"{count} samples")


if __name__ == '__main__':
    main()
