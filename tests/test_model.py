import torch

from models.DiffusionModel import DiffusionModel
from models.Segmentation import SegmentationUNet


def test_diffusion_model():
    model = DiffusionModel(1, 32, 32,256)

    image = torch.randn(4, 1, 32, 32)
    timesteps = torch.randint(1, 1000, (4,))
    pred = model(image, timesteps)

    assert pred.shape == (4, 1, 32, 32)

def test_segmentation_model():
    model = SegmentationUNet(4, 4, 64, 64)

    image = torch.randn(4, 4, 64, 64)
    pred = model(image)

    assert pred.shape == (4, 4, 64, 64)
    

