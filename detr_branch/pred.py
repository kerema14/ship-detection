import torch
from PIL import Image
import torchvision.transforms as T
import os
import random
import cv2
import numpy as np
import json
from pycocotools.coco import COCO

# Define constants
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
GT_COLOR = (0, 255, 0)  # Green for ground truth
finetuned_classes = ['boat']

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load COCO annotations
coco_annotation_path = 'custom/annotations/custom_val.json'
coco = COCO(coco_annotation_path)
# Build mapping from file_name to image_id
file_name_to_id = {img['file_name']: img['id'] for img in coco.loadImgs(coco.getImgIds())}

# Box conversion utilities

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=device)
    return b


# Inference post-processing
def filter_bboxes_from_outputs(outputs, img, threshold=0.7):
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold

    probas_to_keep = probas[keep]
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], img.size)
    return probas_to_keep, bboxes_scaled


# Drawing function
def plot_results_cv2(pil_img, pred_probas=None, pred_boxes=None, gt_boxes=None):
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_h, img_w = img.shape[:2]

    # Draw predicted boxes with thickness proportional to area
    if pred_probas is not None and pred_boxes is not None:
        for p, (xmin, ymin, xmax, ymax), c in zip(pred_probas, pred_boxes.tolist(), COLORS):
            # Compute area
            w = xmax - xmin
            h = ymax - ymin
            area = w * h
            # Thickness scales with sqrt(area)
            thickness = max(1, int(np.sqrt(area) / 30))

            color = tuple([int(i * 255) for i in c])
            cv2.rectangle(img,
                          (int(xmin), int(ymin)),
                          (int(xmax), int(ymax)),
                          color,
                          thickness)
            cl = p.argmax()
            text = f'{finetuned_classes[cl]}: {p[cl]:0.2f}'
            cv2.putText(img,
                        text,
                        (int(xmin), int(ymin) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        thickness)
    
    # Draw ground truth boxes at 0.3 thickness of predictions
    if gt_boxes is not None:
        for (xmin, ymin, w, h) in gt_boxes:
            x2, y2 = xmin + w, ymin + h
            # Approximate GT thickness from a predicted-like thickness
            pred_area = w * h
            base_thick = max(1, int(np.sqrt(pred_area) / 30))
            gt_thick = max(1, int(base_thick * 0.3))

            cv2.rectangle(img,
                          (int(xmin), int(ymin)),
                          (int(x2), int(y2)),
                          GT_COLOR,
                          gt_thick)
            cv2.putText(img,
                        'gt',
                        (int(xmin), int(ymin) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.1,
                        GT_COLOR,
                        gt_thick)

    return img


def combine_images_into_single_image(images, image_size=(500, 500), grid_size=8):
    resized_images = [cv2.resize(img, image_size) for img in images]
    image_w, image_h = image_size
    canvas = np.zeros((image_h * grid_size, image_w * grid_size, 3), dtype=np.uint8)
    for idx, img in enumerate(resized_images):
        row = idx // grid_size
        col = idx % grid_size
        canvas[row * image_h:(row + 1) * image_h,
               col * image_w:(col + 1) * image_w] = img
    return canvas


def run_workflow_on_batch_cv2(image_paths, my_model, output_path="combined_output_images.jpg"):
    images = []
    for img_path in image_paths:
        pil_img = Image.open(img_path).convert('RGB')
        # Get ground truth boxes from COCO
        fname = os.path.basename(img_path)
        img_id = file_name_to_id.get(fname, None)
        gt_boxes = []
        if img_id is not None:
            ann_ids = coco.getAnnIds(imgIds=[img_id])
            anns = coco.loadAnns(ann_ids)
            gt_boxes = [ann['bbox'] for ann in anns]

        # Inference
        img_tensor = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = my_model(img_tensor)
        outputs['pred_logits'] = outputs['pred_logits'].to(device)
        outputs['pred_boxes'] = outputs['pred_boxes'].to(device)
        pred_probas, pred_boxes = filter_bboxes_from_outputs(outputs, pil_img)

        # Draw both pred and GT
        result_img = plot_results_cv2(pil_img, pred_probas, pred_boxes, gt_boxes)
        images.append(result_img)

    combined_image = combine_images_into_single_image(images)
    cv2.imwrite(output_path, combined_image)
    print(f"Combined image saved to {output_path}")

# Load DETR model
model = torch.hub.load('facebookresearch/detr',
                       'detr_resnet50',
                       pretrained=False,
                       num_classes=1)
checkpoint = torch.load('detr/outputs/checkpoint.pth', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model'], strict=False)
model.to(device)
model.eval()

# Run on a random subset of validation images
image_dir = 'custom/val2017/'
all_imgs = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
image_paths = random.sample(all_imgs, 64)
run_workflow_on_batch_cv2(image_paths, model,f"combined_output_images_{str(len(image_paths))}.jpg")
