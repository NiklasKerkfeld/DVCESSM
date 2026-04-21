import torch
from scipy.special import erfinv
from torch import nn


class AdaptGrad(nn.Module):
    def __init__(self, factor: float = 0.2, c: float = .99, x_min: float = -3, x_max: float = 3, N: int = 16):
        super().__init__()
        self.factor = factor
        self.c = c
        self.x_min = x_min
        self.x_max = x_max
        self.N = N

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        denominator = torch.sqrt(torch.tensor(2.0)) * erfinv(self.c)
        sigma = torch.min(self.x_max - x, torch.abs(self.x_min - x)) / denominator

        sigma = torch.where((x <= self.x_min) | (x >= self.x_max), torch.zeros_like(sigma), sigma)

        noise = self.factor * torch.randn((self.N, *x.shape), requires_grad=False) * sigma

        return (x + noise).flatten(0, 1)

   
if __name__ == '__main__':
    from core.Datasets import get_datasets
    import matplotlib.pyplot as plt

    _, test_data = get_datasets('Diffusion', 'data/CelebAMask')
    image = test_data[0]

    adoptgrad = AdaptGrad()

    inputs = adoptgrad(image)

    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].imshow(image.permute(1, 2, 0))
    axarr[0, 1].imshow(inputs[1])
    axarr[1, 0].imshow(inputs[2])
    axarr[1, 1].imshow(inputs[3])

    plt.show()
