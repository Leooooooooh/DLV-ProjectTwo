# postprocess.py

import json
import csv
from collections import defaultdict

PRED_JSON = "pred.json"
PRED_CSV = "pred.csv"
CONF_THRESHOLD = 0.5  # Ignore detections below this

def load_predictions():
    with open(PRED_JSON, "r") as f:
        return json.load(f)

def group_by_image(predictions):
    grouped = defaultdict(list)
    for pred in predictions:
        if pred["score"] >= CONF_THRESHOLD:
            grouped[pred["image_id"]].append(pred)
    return grouped

def process_image_detections(grouped_preds):
    result = []

    for img_id in sorted(grouped_preds.keys()):
        preds = grouped_preds[img_id]

        if not preds:
            result.append((img_id, -1))
            continue

        # Sort by x-coordinate of bbox
        preds_sorted = sorted(preds, key=lambda p: p["bbox"][0])
        digits = [str(p["category_id"]) for p in preds_sorted]

        number_str = "".join(digits) if digits else "-1"
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
    rows = process_image_detections(grouped)
    write_csv(rows)