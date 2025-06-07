#!/usr/bin/env python3
import os
import json
import argparse
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# pip install ultralytics
from ultralytics import YOLO

def build_cat_map(coco_gt):
    # returns list of category_ids sorted by index, so that index i -> cat_id
    cat_ids = sorted(coco_gt.getCatIds())
    return cat_ids

from tqdm import tqdm

import os
import time
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

def evaluate_yolo_on_coco_images(model_path, ann_file, img_dir, conf_threshold=0.001):
    # 1. Load model
    model   = YOLO(model_path)

    # 2. Load COCO GT
    coco_gt = COCO(ann_file)
    cat_map = sorted(coco_gt.getCatIds())

    results = []
    total_time = 0.0
    num_images = 0

    # 3. Inference loop with timing
    for img_id in tqdm(coco_gt.getImgIds(), desc="Running inference"):
        info     = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, info['file_name'])
        if not os.path.exists(img_path):
            continue

        # Time this inference
        start = time.time()
        preds = model.predict(source=img_path,
                              conf=conf_threshold,
                              verbose=False)[0]
        elapsed = time.time() - start

        total_time += elapsed
        num_images += 1
        print(f"Image {info['file_name']} → {elapsed:.3f}s")

        # Collect detections
        for (x1, y1, x2, y2), score, cls_idx in zip(
                preds.boxes.xyxy.cpu().numpy(),
                preds.boxes.conf.cpu().numpy(),
                preds.boxes.cls.cpu().numpy().astype(int)
            ):
            w = x2 - x1
            h = y2 - y1
            results.append({
                "image_id": img_id,
                "category_id": int(cat_map[cls_idx]),
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(score)
            })

    if num_images == 0:
        print("⚠️ No images were processed.")
        return

    avg_time = total_time / num_images
    print(f"\n✔ Processed {num_images} images")
    print(f"✔ Total inference time: {total_time:.3f}s")
    print(f"✔ Average inference time per image: {avg_time:.3f}s")

    # 4. COCO evaluation as before
    coco_dt   = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.imgIds = sorted(coco_gt.getImgIds())
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 inference on COCO images and evaluate via pycocotools"
    )
    parser.add_argument('--model',      default="runs/detect/yolo11n_500px/weights/best.pt",
                        help='Path to your YOLOv8 .pt checkpoint') 
    parser.add_argument('--ann-file',   default='../test/instances_test2017.json',
                        help='Path to COCO annotation JSON (instances_*.json)')
    parser.add_argument('--img-dir',    default="..",
                        help='Directory with the images listed in the JSON')
    parser.add_argument('--conf-threshold', type=float, default=0.001,
                        help='Min confidence for predictions')
    args = parser.parse_args()

    evaluate_yolo_on_coco_images(
        args.model, args.ann_file, args.img_dir, args.conf_threshold
    )
