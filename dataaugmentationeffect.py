import os
import random
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import albumentations as A
import matplotlib.pyplot as plt
from transformers import DetrImageProcessor, DetrForObjectDetection

# —— CONFIG —— 
IMAGE_DIR    = r"C:\Users\kerem\OneDrive\Masaüstü\Politechnika Warszawska\EARIN - Intro to Artificial Intelligence\ship-detection\images"
ANNOT_DIR    = r"C:\Users\kerem\OneDrive\Masaüstü\Politechnika Warszawska\EARIN - Intro to Artificial Intelligence\ship-detection\annotations"
OUTPUT_IMG   = r"C:\Users\kerem\OneDrive\Masaüstü\Politechnika Warszawska\EARIN - Intro to Artificial Intelligence\ship-detection\augmentation_effect.png"
THRESHOLD    = 0.69
GRID_COLS    = 4
GRID_ROWS    = 4
SIZE         = (500, 500)  # size for thumbnails

# 0) Setup model + processor
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = DetrImageProcessor.from_pretrained("fine_tuned_detr_ship_processor")
model     = DetrForObjectDetection.from_pretrained("fine_tuned_detr_ship_model").to(device)
model.eval()
font = ImageFont.load_default()

# 1) Load GT boxes
def load_gt_boxes(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        xmin = float(b.find("xmin").text)
        ymin = float(b.find("ymin").text)
        xmax = float(b.find("xmax").text)
        ymax = float(b.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])
    return boxes

# 2) Define augmentation pipeline (pascal_voc format)
aug = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.Resize(*SIZE)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# 3) Sample images
all_pngs  = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(".png")])
sample_pngs = random.sample(all_pngs, GRID_COLS * GRID_ROWS)

# Prepare grids
orig_thumbs = []
aug_thumbs  = []

for fname in sample_pngs:
    # load
    img_path = os.path.join(IMAGE_DIR, fname)
    xml_path = os.path.join(ANNOT_DIR, fname.replace(".png", ".xml"))
    image = Image.open(img_path).convert("RGB")
    gt_boxes = load_gt_boxes(xml_path)
    
    # inference on original
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    res = processor.post_process_object_detection(outputs, threshold=THRESHOLD, target_sizes=target_sizes)[0]
    orig = image.copy()
    draw = ImageDraw.Draw(orig)
    # draw GT
    for xmin,ymin,xmax,ymax in gt_boxes:
        draw.rectangle([(xmin,ymin),(xmax,ymax)], outline="green", width=2)
    # draw preds
    for box, score in zip(res["boxes"].cpu().tolist(), res["scores"].cpu().tolist()):
        if score>=THRESHOLD:
            draw.rectangle([(box[0],box[1]),(box[2],box[3])], outline="red", width=2)
            draw.text((box[0],box[1]-10), f"{score:.2f}", fill="red", font=font)
    orig_thumbs.append(orig.resize(SIZE))
    
    # apply augmentation to both image and boxes
    arr = np.array(image)
    transformed = aug(image=arr, bboxes=gt_boxes, labels=[1]*len(gt_boxes))
    aug_img = Image.fromarray(transformed['image'])
    aug_boxes = transformed['bboxes']
    
    # inference on augmented
    inputs2 = processor(images=aug_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs2 = model(**inputs2)
    target_sizes2 = torch.tensor([SIZE[::-1]]).to(device)
    res2 = processor.post_process_object_detection(outputs2, threshold=THRESHOLD, target_sizes=target_sizes2)[0]
    aug_vis = aug_img.copy()
    draw2 = ImageDraw.Draw(aug_vis)
    # draw GT on aug
    for xmin,ymin,xmax,ymax in aug_boxes:
        draw2.rectangle([(xmin,ymin),(xmax,ymax)], outline="green", width=2)
    # draw preds on aug
    for box, score in zip(res2["boxes"].cpu().tolist(), res2["scores"].cpu().tolist()):
        if score>=THRESHOLD:
            draw2.rectangle([(box[0],box[1]),(box[2],box[3])], outline="red", width=2)
            draw2.text((box[0],box[1]-10), f"{score:.2f}", fill="red", font=font)
    aug_thumbs.append(aug_vis.resize(SIZE))

# 4) Build combined grid: orig on top, aug below
grid_w = SIZE[0] * GRID_COLS
grid_h = SIZE[1] * GRID_ROWS
combined = Image.new('RGB', (grid_w, grid_h*2))

# paste orig grid
for idx, thumb in enumerate(orig_thumbs):
    row = idx // GRID_COLS
    col = idx % GRID_COLS
    combined.paste(thumb, (col*SIZE[0], row*SIZE[1]))
# paste aug grid
for idx, thumb in enumerate(aug_thumbs):
    row = idx // GRID_COLS
    col = idx % GRID_COLS
    combined.paste(thumb, (col*SIZE[0], grid_h + row*SIZE[1]))

# 5) save and show
combined.save(OUTPUT_IMG)
combined.show()
print(f"Saved augmentation effect grid to {OUTPUT_IMG}")
