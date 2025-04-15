# train.py

import os
import torch
from torch.utils.data import DataLoader
from datasets.coco_digit_dataset import CocoDigitDataset, get_transform
from models.faster_rcnn import get_faster_rcnn_model

def collate_fn(batch):
    return tuple(zip(*batch))

def get_dataloaders(data_dir):
    train_dataset = CocoDigitDataset(
        root=os.path.join(data_dir, "train"),
        annFile=os.path.join(data_dir, "train.json"),
        transforms=get_transform()
    )

    val_dataset = CocoDigitDataset(
        root=os.path.join(data_dir, "valid"),
        annFile=os.path.join(data_dir, "valid.json"),
        transforms=get_transform()
    )

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    return train_loader, val_loader

if __name__ == "__main__":
    data_dir = "datasets/nycu-hw2-data"
    train_loader, val_loader = get_dataloaders(data_dir)

    model = get_faster_rcnn_model(num_classes=11)  # 10 digits + background

    # Next: implement training loop here


