import torch


def norm_distance(image: torch.Tensor, approx_image: torch.Tensor) -> torch.Tensor:
    return torch.norm(image - approx_image, p=2)


def log_distance(image: torch.Tensor, approx_image: torch.Tensor) -> torch.Tensor:
    diff = torch.abs(approx_image - image)
    return torch.sum(torch.log(1 + diff) * 1.4427)


def geman_mcclure_distance(image: torch.Tensor, approx_image: torch.Tensor) -> torch.Tensor:
    # Geman-McClure Distance
    diff = approx_image - image
    gamma = 0.1  # Scale threshold: pixels changing more than gamma incur ~1.0 loss
    return torch.sum(diff ** 2 / (diff ** 2 + gamma ** 2))

def differentiable_soft_l0_count(image: torch.Tensor, approx_image: torch.Tensor) -> torch.Tensor:
    # Differentiable Soft L0 Count
    diff = torch.abs(approx_image - image)
    epsilon = 0.05  # Ignore changes smaller than 5% pixel value
    tau = 20.0  # Steeps the transition into a smooth step function

    return torch.sum(torch.sigmoid(tau * (diff - epsilon)))

def charbonnier_loss(image: torch.Tensor, approx_image: torch.Tensor) -> torch.Tensor:
    # Charbonnier Loss
    diff = approx_image - image
    eps = 1e-3  # Smoothing factor
    return torch.sum(torch.sqrt(diff ** 2 + eps ** 2))