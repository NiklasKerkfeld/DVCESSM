import numpy as np
import torch
from torch.utils.data import DataLoader

from core.Datasets import get_datasets
from models.Segmentation import SegmentationUNet
from utils import calc_iou


def main():
    # check for gpu available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    segmentation = SegmentationUNet(
        in_channels=4,
        out_channels=5,
        img_width=128,
        img_height=128,
        attention=True,
    )

    checkpoint = torch.load("checkpoints/Segmentation/15_SegmentationTraining.pt",
                            map_location=device)
    segmentation.load_state_dict(checkpoint['model_state_dict'])
    segmentation.to(device)

    train_dataset, test_dataset = get_datasets('Segmentation')
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)

    idxs = []
    values = []
    for i, (image, label) in enumerate(dataloader):
        size = label.sum()
        if size > 0:
            image = image.to(device)
            label = label.to(device)

            with torch.no_grad():
                pred = segmentation(image).argmax(dim=1)

            iou = calc_iou(label, pred)
            print(i, iou)
            idxs.append(i)
            values.append(iou.item())

    print(f"\n\nresults:")
    top10 = np.argsort(np.array(values))[:10]
    for i in top10:
        print(idxs[i], values[i])
    
   
if __name__ == '__main__':
    main()
