# models/faster_rcnn.py

import timm
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection.anchor_utils import AnchorGenerator
from models.custom_roi_head import CustomFastRCNNPredictor
import torchvision

class TimmBackbone(nn.Module):
    def __init__(self, model_name="seresnet50", pretrained=True, out_channels=256):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        self.out_channels = out_channels

        self.conv = nn.Conv2d(self.backbone.feature_info[-1]["num_chs"], out_channels, kernel_size=1)

    def forward(self, x):
        feats = self.backbone(x)
        # Use only the last feature map for now
        return {"0": self.conv(feats[-1])}

def get_faster_rcnn_model(num_classes: int):
    backbone = TimmBackbone()

    # Create anchor generator manually (can tune this)
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0"],
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor.cls_score = CustomFastRCNNPredictor(in_features, num_classes)

    return model