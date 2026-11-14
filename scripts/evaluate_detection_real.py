#!/usr/bin/env python3
"""
EduSense Real-Image Detection & Latency Benchmark
==================================================
Evaluates StrictPersonFaceDetector + EduSensePipeline on real COCO val2017 images
(person-present images vs empty/non-person/screen images) and measures:
- Person Detection Precision, Recall, F1-score, False Positive Rate, Missed Detection Rate
- Validated Face Detection count
- Mean inference latency (ms) and processing FPS
"""

import json
import sys
import time
from pathlib import Path
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from inference.pipeline import EduSensePipeline
from inference.tracker import compute_iou

COCO_ANN = BASE_DIR / "datasets" / "COCO" / "annotations" / "instances_val2017.json"
COCO_IMG_DIR = BASE_DIR / "datasets" / "COCO" / "val2017"


def main():
    if not COCO_ANN.exists() or not COCO_IMG_DIR.exists():
        print("COCO val2017 not found; skipping empirical COCO benchmark.")
        return

    data = json.loads(COCO_ANN.read_text(encoding="utf-8"))
    img_info = {img["id"]: img["file_name"] for img in data["images"]}
    person_boxes_by_img = {}
    all_cat_by_img = {}

    for ann in data["annotations"]:
        img_id = ann["image_id"]
        all_cat_by_img.setdefault(img_id, set()).add(ann["category_id"])
        if ann["category_id"] == 1 and not ann.get("iscrowd", 0):
            x, y, w, h = ann["bbox"]
            if w >= 36 and h >= 50:
                person_boxes_by_img.setdefault(img_id, []).append((x, y, x + w, y + h))

    pos_ids = [iid for iid, boxes in person_boxes_by_img.items() if 1 <= len(boxes) <= 5][:25]
    neg_ids = [iid for iid, cats in all_cat_by_img.items() if 1 not in cats][:15]

    pipeline = EduSensePipeline(mode="webcam", open_camera=False)
    tp = fp = fn = 0
    neg_fp_images = 0
    latencies = []
    total_valid_faces = 0

    for iid in pos_ids:
        img = cv2.imread(str(COCO_IMG_DIR / img_info[iid]))
        if img is None:
            continue
        pipeline.reset_tracking()
        t0 = time.perf_counter()
        _, res = pipeline.process_frame_array(img, camera_source="EVAL_POS", encode_jpeg=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        gt_boxes = person_boxes_by_img[iid]
        # Rescale GT boxes if image was resized to FRAME_MAX_W
        oh, ow = img.shape[:2]
        from inference.pipeline import FRAME_MAX_W
        scale = min(1.0, FRAME_MAX_W / float(ow))
        gt_scaled = [(b[0]*scale, b[1]*scale, b[2]*scale, b[3]*scale) for b in gt_boxes]

        pred_boxes = [tuple(s["person_bbox"]) for s in res["students"] if s.get("person_bbox")]
        total_valid_faces += res["valid_face_count"]

        matched_gt = set()
        for pb in pred_boxes:
            best_iou, best_j = 0.0, -1
            for j, gb in enumerate(gt_scaled):
                if j in matched_gt:
                    continue
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= 0.35 and best_j != -1:
                tp += 1
                matched_gt.add(best_j)
            else:
                fp += 1
        fn += len(gt_scaled) - len(matched_gt)

    for iid in neg_ids:
        img = cv2.imread(str(COCO_IMG_DIR / img_info[iid]))
        if img is None:
            continue
        pipeline.reset_tracking()
        t0 = time.perf_counter()
        _, res = pipeline.process_frame_array(img, camera_source="EVAL_NEG", encode_jpeg=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        if res["live_detected_persons"] > 0 or res["valid_face_count"] > 0:
            neg_fp_images += 1
            fp += res["live_detected_persons"]

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-9, precision + recall)
    fpr_neg = neg_fp_images / max(1, len(neg_ids))
    miss_rate = fn / max(1, tp + fn)
    mean_lat = sum(latencies) / max(1, len(latencies))
    fps = 1000.0 / max(1e-6, mean_lat)

    print("=== EduSense Real COCO val2017 Evaluation (25 Person + 15 Non-Person Images) ===")
    print(f"TP={tp}, FP={fp}, FN={fn}, ValidatedFaces={total_valid_faces}")
    print(f"Precision:          {precision:.3f}")
    print(f"Recall:             {recall:.3f}")
    print(f"F1-Score:           {f1:.3f}")
    print(f"Missed Det Rate:    {miss_rate:.3f}")
    print(f"Empty/Neg FPR:      {fpr_neg:.3f} ({neg_fp_images}/{len(neg_ids)} images)")
    print(f"Mean Latency (ms):  {mean_lat:.1f}")
    print(f"Processing FPS:     {fps:.1f}")


if __name__ == "__main__":
    main()

