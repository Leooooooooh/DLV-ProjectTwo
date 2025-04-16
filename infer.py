# infer.py

import os
import json
import torch
from PIL import Image
from torchvision import transforms
from models.faster_rcnn import get_faster_rcnn_model

# ========= CONFIG ========= #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 11  # digits 0-9 + background
IMAGE_DIR = "datasets/nycu-hw2-data/test"
CKPT_PATH = "checkpoints/fasterrcnn_epoch_2.pth"
OUTPUT_JSON = "pred.json"
SCORE_THRESHOLD = 0.5
# ========================== #

# Preprocessing transform
transform = transforms.Compose([
    transforms.ToTensor()
])

def load_image(img_path):
    image = Image.open(img_path).convert("RGB")
    return transform(image)

def load_model():
    model = get_faster_rcnn_model(NUM_CLASSES)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def run_inference(model):
    results = []
    image_list = sorted(os.listdir(IMAGE_DIR))

    with torch.no_grad():
        for img_name in image_list:
            if not img_name.endswith((".png", ".jpg", ".jpeg")):
                continue

            img_id = int(os.path.splitext(img_name)[0])
            img_path = os.path.join(IMAGE_DIR, img_name)
            image = load_image(img_path).to(DEVICE).unsqueeze(0)

            output = model(image)[0]
            boxes = output["boxes"]
            labels = output["labels"]
            scores = output["scores"]

            for box, label, score in zip(boxes, labels, scores):
                if score < SCORE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = box.tolist()
                w = x2 - x1
                h = y2 - y1

                results.append({
                    "image_id": img_id,
                    "category_id": label.item(),
                    "bbox": [x1, y1, w, h],
                    "score": score.item()
                })

    return results

if __name__ == "__main__":
    model = load_model()
    predictions = run_inference(model)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(predictions, f)
    print(f"[✓] Saved predictions to {OUTPUT_JSON}")