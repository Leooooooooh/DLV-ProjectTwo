from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

def get_faster_rcnn_model(num_classes: int):
    # Load model with pretrained backbone
    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the head
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model