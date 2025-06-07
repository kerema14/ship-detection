#!/usr/bin/env python3
import os
import json
import glob
import argparse

def convert_coco_to_yolo(coco_root):
    """
    For each COCO annotation JSON in coco_root/annotations,
    create YOLO-format label files under coco_root/labels/<split>/.
    """
    ann_dir    = os.path.join(coco_root, 'annotations')
    labels_root = os.path.join(coco_root, 'labels')
    os.makedirs(labels_root, exist_ok=True)

    # Process every 'instances_*.json' file
    for ann_path in glob.glob(os.path.join(ann_dir, 'instances_*.json')):
        split = os.path.splitext(os.path.basename(ann_path))[0].replace('instances_', '')
        out_dir = os.path.join(labels_root, split)
        os.makedirs(out_dir, exist_ok=True)

        with open(ann_path, 'r') as f:
            data = json.load(f)

        # Build lookup: image_id -> {file_name, width, height}
        images = {img['id']: img for img in data['images']}

        # Build mapping category_id -> consecutive class index [0..num_cats-1]
        cats = data.get('categories', [])
        cat_ids = sorted(c['id'] for c in cats)
        cat_to_idx = {cid: idx for idx, cid in enumerate(cat_ids)}

        # Convert each annotation
        for ann in data['annotations']:
            img_info = images[ann['image_id']]
            iw, ih = img_info['width'], img_info['height']
            x, y, w, h = ann['bbox']

            # Convert to YOLO format (normalized center x, center y, width, height)
            x_center = (x + w / 2) / iw
            y_center = (y + h / 2) / ih
            w_norm   = w / iw
            h_norm   = h / ih

            cls_idx = cat_to_idx[ann['category_id']]

            # Write to label file named like the image (but .txt)
            base = os.path.splitext(img_info['file_name'])[0]
            label_path = os.path.join(out_dir, f"{base}.txt")
            with open(label_path, 'a') as lf:
                lf.write(f"{cls_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        print(f"→ Converted {split}, labels saved in {out_dir}")

if __name__ == "__main__":
    coco_root = "C:/Users/kerem/OneDrive/Masaüstü/Politechnika Warszawska/EARIN - Intro to Artificial Intelligence/ship-detection/retinanet_branch/coco"
    
    convert_coco_to_yolo(coco_root=coco_root)
