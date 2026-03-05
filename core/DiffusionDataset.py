from torch.utils.data import Dataset

class DiffusionDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __getitem__(self, index):
        image, _ = self.base_dataset[index]  # Get image/label, discard label
        return image

    def __len__(self):
        return len(self.base_dataset)

