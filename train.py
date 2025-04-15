import os
import time
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

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)

    return train_loader, val_loader

def train_model(model, train_loader, val_loader, device, num_epochs=10, lr=0.005, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start = time.time()

        for i, (images, targets) in enumerate(train_loader):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

            if (i + 1) % 50 == 0 or (i + 1) == len(train_loader):
                print(f"[Epoch {epoch+1} | Batch {i+1}/{len(train_loader)}] Loss: {losses.item():.4f}")

        scheduler.step()

        print(f"\n[Epoch {epoch+1} DONE] Total Loss: {epoch_loss:.4f} | Time: {time.time() - start:.2f}s")

        ckpt_path = os.path.join(save_dir, f"fasterrcnn_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to: {ckpt_path}\n")


if __name__ == "__main__":
    data_dir = "datasets/nycu-hw2-data"
    train_loader, val_loader = get_dataloaders(data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_faster_rcnn_model(num_classes=11)

    train_model(model, train_loader, val_loader, device, num_epochs=10, lr=0.005)