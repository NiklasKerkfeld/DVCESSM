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



