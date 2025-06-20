# 🚢 Ship Object Detection

This project was developed for the *Introduction to Artificial Intelligence* course at Warsaw University of Technology (Summer 2025). The goal is to detect ships and boats from aerial images using multiple deep learning object detection architectures.

## 📦 Models Implemented

Three object detection architectures were implemented and compared:

- **YOLOv8 (Medium)** – Ultralytics PyTorch implementation
- **RetinaNet** – ResNet-50 backbone with COCO weights
- **DETR** – Transformer-based detection model with ResNet-50 backbone

## 🧠 Training and Evaluation

| Model       | mAP@0.5 | mAP@[0.5:0.95] | Small | Medium | Large | Inference Speed |
|-------------|---------|----------------|-------|--------|--------|-----------------|
| RetinaNet   | 77.7%   | 51.5%          | 19.1% | 73.1%  | 83.6%  | 45–60 ms/image  |
| DETR        | 83.3%   | 55.0%          | 19.2% | 77.5%  | 90.1%  | 30–45 ms/image  |
| YOLOv8 (M)  | **90.6%** | **61.0%**    | **31.5%** | **80.0%** | 87.5% | 36 ms/image     |

## 🧪 Dataset

- **Source**: [Ship Detection Dataset on Kaggle](https://www.kaggle.com/datasets/andrewmvd/ship-detection)
- **Format**: PASCAL VOC (converted to COCO and YOLO where needed)
- **Classes**: Single class - `boat`
- **Total Images**: 621
- **Splits**: 70% train / 15% val / 15% test

## 🛠️ Tools and Frameworks

- PyTorch 2.3.1 + cu121
- Ultralytics YOLOv8
- forked yhenon’s RetinaNet fork
- forked DETR (Facebook AI Research)
- OpenCV, Seaborn, Matplotlib
- GitHub for version control

## 📊 Evaluation Metrics

- `mAP@0.5`, `mAP@0.5:0.95`
- Inference time per image
- Size-wise mAP (small, medium, large)

## 🔍 Observations & Insights

- **YOLOv8** performed best overall, especially with small object detection due to heavy augmentations (Mosaic, MixUp, Copy-Paste, etc.).
- **DETR** offered balanced performance and fast inference.
- **RetinaNet** had strong results for medium/large boats but struggled with small and densely clustered objects.

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/ship-object-detection.git
cd ship-object-detection
```

### 2. Set up virtual environment
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Train a model
YOLOv8:
use the `train.py` script in yolo_branch
or
```bash
cd yolo_branch
yolo detect train \
  model=yolov8m.pt \
  data=config.yaml \
  epochs=150 \
  imgsz=512 \
  patience=35 \
  batch=-1 \
  device=0 \
  seed=42 \
  lr0=0.0001 \
  weight_decay=0.0005
```


DETR:
first, run `detr_branch/get_checkpoint.py` to get the headless checkpoint, 
then use the `train.py` script in detr_branch
or
```bash
cd detr_branch/detr
python main.py \
  --dataset_file custom \
  --coco_path ../custom/ \
  --output_dir outputs \
  --resume detr-r50_no-class-head.pth \
  --num_classes 1 \
  --epochs 300 \
  --lr_drop 50 \
  --lr_drop_rate 0.59234
```


RetinaNet:
first, run `retinanet_branch/get_checkpoint.py` to get the headless checkpoint, 
then use the train.py script in detr_branch
use the `train_process.py` script in retinanet_branch
or
```bash
cd retinanet_branch/pytorch-retinanet
python train.py \
  --dataset coco \
  --coco_path ../coco \
  --epochs 300 \
  --depth 50
```


### 4. Run inference
See the `/runs/weights` for yolo, `/outputs` for detr and retinanet to see the outputted pytorch models after and during the training.
run pred.py from each of the branches to see some results. or you can write your own inference script since all of the models are in pytorch format.

## 📄 Report

You can find the detailed final report in `report/Ship_Detection__Final.pdf`, covering:

- Dataset analysis
- Model architecture details
- Training setups
- Results and failure analysis
- Performance comparison
- Future suggestions (e.g. sliding-window for small object detection)

## 📌 Credits

- Developed by **Kerem Adalı** for AI course project @ PW
- Instructor: Dr. Muhammad Farhan

## 📚 References

- [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
- [DETR: End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
- [RetinaNet (Focal Loss for Dense Detection)](https://arxiv.org/abs/1708.02002)
