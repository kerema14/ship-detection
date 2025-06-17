import os
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from ultralytics import YOLO


def draw_boxes(img, boxes, labels, colors, thickness=2, font_scale=0.25):
    """
    Draws bounding boxes with labels on an image.
    boxes: list of [x1, y1, x2, y2]
    labels: list of strings
    colors: list of (B, G, R) tuples
    """
    for box, label, color in zip(boxes, labels, colors):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if label:
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, 1, cv2.LINE_AA)
    return img


def main(model_path, ann_file, img_dir, output_path,
         num_samples=64, conf_threshold=0.25, tile_size=512, seed=42):
    # Set random seed
    random.seed(seed)

    # Load model and COCO annotations
    model = YOLO(model_path)
    coco = COCO(ann_file)

    # Map category IDs to names
    cats = coco.loadCats(coco.getCatIds())
    cat_map = {cat['id']: cat['name'] for cat in cats}

    # Get image IDs and sample
    img_ids = coco.getImgIds()
    sampled_ids = random.sample(img_ids, min(num_samples, len(img_ids)))

    # Prepare output grid
    grid_cols = int(np.ceil(np.sqrt(len(sampled_ids))))
    grid_rows = int(np.ceil(len(sampled_ids) / grid_cols))
    grid_w = grid_cols * tile_size
    grid_h = grid_rows * tile_size
    canvas = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)

    # Iterate and draw
    for idx, img_id in enumerate(sampled_ids):
        info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, info['file_name'])
        img = cv2.imread(img_path)
        if img is None:
            continue
        # Resize for tile
        img = cv2.resize(img, (tile_size, tile_size))

        # Draw ground truth
        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        gt_boxes = []
        gt_labels = []
        gt_colors = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            # scale to tile size
            x1 = x * tile_size / info['width']
            y1 = y * tile_size / info['height']
            x2 = (x + w) * tile_size / info['width']
            y2 = (y + h) * tile_size / info['height']
            gt_boxes.append([x1, y1, x2, y2])
            gt_labels.append("")
            gt_colors.append((0, 255, 0))  # green for GT
        # Run inference
        preds = model.predict(source=img, conf=conf_threshold, imgsz=tile_size, verbose=False)[0]
        boxes = preds.boxes.xyxy.cpu().numpy()
        scores = preds.boxes.conf.cpu().numpy()
        classes = preds.boxes.cls.cpu().numpy().astype(int)
        img = draw_boxes(img, gt_boxes, gt_labels, gt_colors, thickness=3)

        

        # Draw predictions
        pred_boxes = []
        pred_labels = []
        pred_colors = []
        for (x1, y1, x2, y2), score, cls_idx in zip(boxes, scores, classes):
            pred_boxes.append([x1, y1, x2, y2])
            label = f"{model.names[cls_idx]}:{score:.2f}"
            pred_labels.append(label)
            pred_colors.append((0, 0, 255))  # red for preds
        img = draw_boxes(img, pred_boxes, pred_labels, pred_colors, thickness=2)

        # Paste into canvas
        row = idx // grid_cols
        col = idx % grid_cols
        y0 = row * tile_size
        x0 = col * tile_size
        canvas[y0:y0+tile_size, x0:x0+tile_size] = img

    # Save output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, canvas)
    print(f"Saved visualization grid to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualize COCO GT and YOLO predictions on a grid of images"
    )
    parser.add_argument('--model',      default="runs/detect/yolo11n_500px/weights/best.pt",
                        help='Path to your YOLOv8 .pt checkpoint') 
    parser.add_argument('--ann-file',   default='../retinanet_branch/coco/annotations/instances_val2017.json',
                        help='Path to COCO annotation JSON (instances_*.json)')
    parser.add_argument('--img-dir',    default="../retinanet_branch/coco/images/val2017/",
                        help='Directory with the images listed in the JSON')
    parser.add_argument('--output',     default="grid_64_val_2.jpg", help='Output file path (e.g. grid.jpg)')
    parser.add_argument('--num-samples', type=int, default=64,
                        help='Number of images to sample')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                        help='Confidence threshold for predictions')
    parser.add_argument('--tile-size',  type=int, default=512,
                        help='Size of each tile in the grid')
    parser.add_argument('--seed',       type=int, default=0,
                        help='Random seed for sampling')
    args = parser.parse_args()
    
    main(args.model, args.ann_file, args.img_dir, args.output,
         num_samples=args.num_samples,
         conf_threshold=args.conf_threshold,
         tile_size=args.tile_size,
         seed=args.seed)
