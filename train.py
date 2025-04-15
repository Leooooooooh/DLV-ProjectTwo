import torch
import torchvision
import os
import time
from datasets.coco_digit_dataset import CocoDigitDataset, get_transform
from models.faster_rcnn import get_faster_rcnn_model
from torch.utils.data import DataLoader

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

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)

    return train_loader, val_loader


def train_model(model, train_loader, val_loader, device, num_epochs=10, lr=0.005):
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start = time.time()

        for images, targets in train_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        lr_scheduler.step()

        print(f"[Epoch {epoch+1}] Loss: {epoch_loss:.4f} | Time: {time.time() - start:.2f}s")

        # Save checkpoint
        ckpt_path = f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    data_dir = "datasets/nycu-hw2-data"
    train_loader, val_loader = get_dataloaders(data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_faster_rcnn_model(num_classes=11)

    train_model(model, train_loader, val_loader, device, num_epochs=10, lr=0.005)