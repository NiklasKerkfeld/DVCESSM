import torch

def random_target(input_tensor, n):
    offset = torch.randint(1, n - 1, input_tensor.shape, device=input_tensor.device)
    target = ((input_tensor - 1 + offset) % (n - 1)) + 1
    return target


def get_one_of_each(dataset):
    samples = {}

    # Iterate through the dataset
    for i in range(len(dataset)):
        image, label = dataset[i]
        # If we haven't found this digit yet, save it
        if label not in samples:
            samples[label] = image

        # Stop once we have all 10 digits (0-9)
        if len(samples) == 10:
            break

    # Sort by label to ensure they are in order 0-9
    return torch.cat([samples[i] for i in range(10)])[:, None]


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)

    # Calculate alpha_bar (cumprod)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    # Derive betas from alphas_cumprod
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    # Clamp betas to prevent singularities (max value of 0.999 is standard)
    return torch.clip(betas, 0.0001, 0.9999)


def to_vis(x):
    return torch.clamp((x + 1) / 2, 0.0, 1.0)
