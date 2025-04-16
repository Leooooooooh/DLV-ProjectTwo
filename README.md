# Digit Recognition with Faster R-CNN

This project tackles the dual task of digit localization and sequence recognition using a modified Faster R-CNN model with a SE-ResNet50 backbone.

---

## 📁 Project Structure
ProjectRoot/
├── datasets/
│   └── nycu-hw2-data/
│       ├── train/
│       ├── valid/
│       ├── test/
│       ├── train.json
│       └── valid.json
│
├── models/
│   ├── faster_rcnn.py            # Defines model architecture
│   └── custom_roi_head.py        # Custom RoI head with dropout and FC layer
│
├── datasets/
│   └── coco_digit_dataset.py     # Custom dataset loader for COCO-format digits
│
├── train.py                      # Main training script
├── infer.py                      # Inference script to generate pred.json
├── postprocess.py                # Converts pred.json to pred.csv for Task 2
├── checkpoints/                  # Folder for saved model weights
│   └── checkpoint_epoch_*.pth
│
├── pred.json                     # Output from inference (Task 1)
├── pred.csv                      # Output for Task 2 (formatted predictions)
└── README.md                     # This file

## Usage
### 1. Install Dependencies

```bash
pip install torch torchvision timm pycocotools safetensors
```
### 2. train the model
```bash
python3 train.py
```

### 3. run inference (task1)
```bash
python3 infer.py
```

### 4. generate output (task2)
```bash
python3 postprocess.py
```
