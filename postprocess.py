# postprocess.py

import json
import csv
import os
from collections import defaultdict

PRED_JSON = "pred.json"
PRED_CSV = "pred.csv"
IMAGE_DIR = "datasets/nycu-hw2-data/test"  # adjust if needed
CONF_THRESHOLD = 0.5

def load_predictions():
    with open(PRED_JSON, "r") as f:
        return json.load(f)

def get_all_image_ids():
    filenames = sorted(os.listdir(IMAGE_DIR))
    image_ids = [int(os.path.splitext(fname)[0]) for fname in filenames if fname.endswith(".png")]
    return sorted(image_ids)

def group_by_image(predictions):
    grouped = defaultdict(list)
    for pred in predictions:
        if pred["score"] >= CONF_THRESHOLD:
            grouped[pred["image_id"]].append(pred)
    return grouped

def process_all_images(all_image_ids, grouped_preds):
    result = []

    for img_id in all_image_ids:
        preds = grouped_preds.get(img_id, [])

        if not preds:
            result.append((img_id, -1))
            continue

        preds_sorted = sorted(preds, key=lambda p: p["bbox"][0])
        digits = [str(p["category_id"] - 1) for p in preds_sorted]
        number_str = "".join(digits).lstrip("0") or "0"
        result.append((img_id, number_str))

    return result

def write_csv(rows):
    with open(PRED_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "pred_label"])
        writer.writerows(rows)

    print(f"[✓] Saved {len(rows)} predictions to {PRED_CSV}")

if __name__ == "__main__":
    predictions = load_predictions()
    grouped = group_by_image(predictions)
    all_image_ids = get_all_image_ids()
    rows = process_all_images(all_image_ids, grouped)
    write_csv(rows)