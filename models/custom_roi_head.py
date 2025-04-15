# models/custom_roi_head.py

import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class CustomFastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, 1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        scores = self.fc2(x)
        return scores