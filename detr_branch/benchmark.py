import time
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from tqdm import tqdm
from torchvision import transforms as T
from PIL import Image

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('facebookresearch/detr',
                       'detr_resnet50',
                       pretrained=False,
                       num_classes=1)

# Load checkpoint and move the model to the correct device
checkpoint = torch.load(r'C:\Users\kerem\OneDrive\Masaüstü\Politechnika Warszawska\EARIN - Intro to Artificial Intelligence\ship-detection\detr_branch\detr\saved_outputs\outputs\checkpoint.pth', 
                       map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model'], strict=False)
model.to(device)
model.eval()

# Load the COCO dataset and annotations
coco_gt = COCO('test/test.json')
coco_dt = []

# Get image ids
img_ids = coco_gt.getImgIds()

def box_cxcywh_to_xyxy(x):
    """Convert boxes from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2)"""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def rescale_bboxes(out_bbox, size):
    """Rescale normalized bounding boxes to image size"""
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=out_bbox.device)
    return b


print("Categories in ground truth:")
cats = coco_gt.loadCats(coco_gt.getCatIds())
for cat in cats:
    print(f"ID: {cat['id']}, Name: {cat['name']}")

print(f"\nProcessing {len(img_ids)} images...")

times = []
# Iterate over images in the dataset
for img_id in tqdm(img_ids):
    img_info = coco_gt.loadImgs(img_id)[0]
    img_path = img_info['file_name']
    
    try:
        # Load image and get original size
        img = Image.open(img_path).convert('RGB')
        orig_size = img.size  # (width, height)
        
        # Transform for model
        img_tensor = transform(img).unsqueeze(0).to(device)
        starttime = time.time()
        # Get model predictions
        with torch.no_grad():
            outputs = model(img_tensor)
        endtime = time.time()
        times.append(endtime - starttime)
        # Extract predictions
        pred_logits = outputs['pred_logits'][0]  # Remove batch dimension [num_queries, num_classes]
        pred_boxes = outputs['pred_boxes'][0]    # Remove batch dimension [num_queries, 4]

        # Convert logits to probabilities
        prob = pred_logits.softmax(-1)
        
        # For DETR with num_classes=1, we have [background, ship] classes
        # Background is typically the last class in DETR
        if prob.shape[-1] == 2:  # [background, ship]
            # Get ship class scores (class index 0, since background is at index 1)
            scores = prob[:, 0]  # Ship class probabilities
            labels = torch.zeros(len(scores), dtype=torch.long, device=device)  # All are ship class (0)
        else:
            # If we have different setup, get max probability
            scores, labels = prob[..., :-1].max(-1)  # Exclude background class
        
        # Apply confidence threshold
        threshold = 0.05  # Start with lower threshold
        keep = scores >= threshold
        
        if keep.sum() > 0:  # Only process if we have detections
            # Filter predictions
            valid_scores = scores[keep]
            valid_labels = labels[keep] 
            valid_boxes = pred_boxes[keep]
            
            # Convert normalized boxes to absolute coordinates
            boxes_abs = rescale_bboxes(valid_boxes, orig_size)
            
            # Convert to COCO format [x, y, width, height] and move to CPU
            for i in range(len(boxes_abs)):
                x1, y1, x2, y2 = boxes_abs[i].cpu().numpy()
                
                # Ensure coordinates are valid
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                
                if width > 0 and height > 0:  # Only add valid boxes
                    # Make sure category_id matches wer ground truth
                    # Check wer test.json to see what category ID ships have
                    category_id = cats[0]['id'] if cats else 1  # Use first category from ground truth
                    
                    coco_dt.append({
                        'image_id': int(img_id),
                        'category_id': int(category_id),
                        'bbox': [x1, y1, width, height],
                        'score': float(valid_scores[i].cpu().numpy())
                    })
    
    except Exception as e:
        print(f"Error processing image {img_id}: {e}")
        continue

print(f"\nTotal detections: {len(coco_dt)}")

if len(coco_dt) == 0:
    print("No detections found! Try:")
    print("1. Lowering the confidence threshold")
    print("2. Checking if wer model is loaded correctly")
    print("3. Verifying image paths in test.json")
else:
    # Show some sample detections
    print("\nSample detections:")
    for i in range(min(5, len(coco_dt))):
        det = coco_dt[i]
        print(f"Image {det['image_id']}: bbox={det['bbox']}, score={det['score']:.3f}")
    
    try:
        # Load results and evaluate
        coco_dt_obj = coco_gt.loadRes(coco_dt)
        coco_eval = COCOeval(coco_gt, coco_dt_obj, 'bbox')
        
        # we can specify which images and categories to evaluate
        # coco_eval.params.imgIds = img_ids  # Optional: specify image IDs
        # coco_eval.params.catIds = [1]      # Optional: specify category IDs
        
        print("\nRunning evaluation...")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Print detailed stats
        print(f"\nDetailed Results:")
        print(f"Average Precision (AP) @ IoU=0.50:0.95: {coco_eval.stats[0]:.4f}")
        print(f"Average Precision (AP) @ IoU=0.50: {coco_eval.stats[1]:.4f}")
        print(f"Average Precision (AP) @ IoU=0.75: {coco_eval.stats[2]:.4f}")
        print(f"Average inference time per image: {np.mean(times)*1000} ms")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("This might be due to format issues or no matching detections")