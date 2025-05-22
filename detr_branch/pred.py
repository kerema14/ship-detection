import torch
from PIL import Image
import torchvision.transforms as T
import os
import random
import cv2
import numpy as np

# Define constants
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
finetuned_classes = ['boat']

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=device)  # Ensure tensor is on the right device
    return b

def filter_bboxes_from_outputs(outputs, img, threshold=0.7):
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold

    probas_to_keep = probas[keep]
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], img.size)
    
    return probas_to_keep, bboxes_scaled

def plot_finetuned_results_cv2(pil_img, prob=None, boxes=None):
    img = np.array(pil_img)  # Convert to numpy array for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
    
    if prob is not None and boxes is not None:
        for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS):
            color = tuple([int(i * 255) for i in c])  # Convert float to integer and multiply by 255
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
            cl = p.argmax()
            text = f'{finetuned_classes[cl]}: {p[cl]:0.2f}'
            cv2.putText(img, text, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    return img

def combine_images_into_single_image(images, image_size=(500, 500)):
    # Resize images to a consistent size (image_size)
    resized_images = [cv2.resize(img, image_size) for img in images]

    # Create a blank canvas for the final image (6x6 grid for 36 images)
    grid_size = 8  # 6x6 grid for 36 images
    image_width, image_height = image_size
    
    combined_image = np.zeros((image_height * grid_size, image_width * grid_size, 3), dtype=np.uint8)
    
    # Place each image into the appropriate position on the canvas
    for idx, img in enumerate(resized_images):
        row = idx // grid_size
        col = idx % grid_size
        combined_image[row * image_height: (row + 1) * image_height,
                       col * image_width: (col + 1) * image_width] = img
    
    return combined_image

def run_workflow_on_batch_cv2(image_paths, my_model, output_path="combined_output_images.jpg"):
    images = []
    
    for i, img_path in enumerate(image_paths):
        # Open image and ensure it's in RGB mode
        img = Image.open(img_path).convert('RGB')

        # Prepare image and move it to the device (GPU or CPU)
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():  # No need to compute gradients
            outputs = my_model(img_tensor)

        # Ensure all outputs are on the correct device
        outputs['pred_logits'] = outputs['pred_logits'].to(device)
        outputs['pred_boxes'] = outputs['pred_boxes'].to(device)

        # Filter bounding boxes and predictions
        probas_to_keep, bboxes_scaled = filter_bboxes_from_outputs(outputs, img, threshold=0.7)

        # Get processed image with bounding boxes
        processed_img = plot_finetuned_results_cv2(img, probas_to_keep, bboxes_scaled)
        images.append(processed_img)

    # Combine the images into a single large image
    combined_image = combine_images_into_single_image(images)

    # Save or display the final image
    cv2.imwrite(output_path, combined_image)
    print(f"Combined image saved to {output_path}")

# Load model
model = torch.hub.load('facebookresearch/detr',
                       'detr_resnet50',
                       pretrained=False,
                       num_classes=1)

# Load checkpoint and move the model to the correct device
checkpoint = torch.load('outputs/checkpoint.pth', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model'], strict=False)
model.to(device)
model.eval()

# Get list of images from a directory
image_dir = 'C:/Users/kerem/OneDrive/Masaüstü/Politechnika Warszawska/EARIN - Intro to Artificial Intelligence/ship-detection/detr_branch/custom/val2017/'
image_paths = random.sample([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')], 64)

# Run workflow on 36 random images using OpenCV
run_workflow_on_batch_cv2(image_paths, model)
