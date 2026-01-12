from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class DiscriminatorDataset(Dataset):
    def __init__(self, real_images):
        super(DiscriminatorDataset, self).__init__()
        self.factor = 0
        self.real_images = real_images
        self.fake_images = []


    def __getitem__(self, idx):
        if idx < (len(self.real_images) * self.factor):
            return self.real_images[idx % len(self.real_images)], 0
        else:
            return self.fake_images[idx - (len(self.real_images) * self.factor)], 1

    def __len__(self):
        return (2 *  self.factor) * len(self.real_images)

    def add_fakes(self, fake_images):
        assert(len(fake_images) == len(self.real_images))

        self.fake_images.extend(fake_images)
        self.factor += 1
    
   
if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Pad(2, 0.0)])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    images = [train_dataset[i][0] for i in range(len(train_dataset))]

    dataset = DiscriminatorDataset(images)
    dataset.add_fakes(fake_images=images)
    dataset.add_fakes(fake_images=images)

    print(len(dataset))
    image, label = dataset[0]
    print(image.shape, label)

    image, label = dataset[180000]
    print(image.shape, label)

    image, label = dataset[240000]
    print(image.shape, label)
