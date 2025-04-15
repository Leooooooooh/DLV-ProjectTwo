# datasets/coco_digit_dataset.py

import torch
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
import os

class CocoDigitDataset(CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super().__init__(root, annFile)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)
        boxes = []
        labels = []

        for ann in anns:
            x_min, y_min, w, h = ann['bbox']
            boxes.append([x_min, y_min, x_min + w, y_min + h])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
        }

        if self._transforms:
            img = self._transforms(img)

        return img, target

def get_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToTensor()
    ])