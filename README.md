# Intelligent-Landmark-Detection-System
Localizing and classifying real-world landmarks in unconstrained photographs
A deep-learning–based system for detecting, localizing, and classifying real-world landmarks in unconstrained photographs.
The project compares a custom-built object detection model against a Faster R-CNN baseline, evaluates performance, and visualizes predictions using annotated bounding boxes.


## 📌 Features

- 🔧 **Custom Object Detector** trained from scratch  
- ⚡ **Faster R-CNN Benchmark** for comparison  
- 🖼️ Support for **COCO JSON** and **Pascal VOC XML** formats  
- 🎯 Handles **normalized** (custom) and **pixel-based** (Faster R-CNN) bounding boxes  
- 📊 Generates annotated images with confidence scores  
- 🔁 Fully reproducible training using fixed seeds and deterministic pipeline  
- 🧪 Evaluation metrics including **mAP**, **Precision**, **Recall**, and **FPS**  

---

## 📂 Project Structure

```text
Intelligent-Landmark-Detection-System/
│── configs/
│   ├── custom.yaml
│   └── classes.txt
│── data/
│   ├── train/
│   ├── val/
│   └── test/
│── models/
│── outputs/
│   ├── checkpoints/
│   └── results/
│── train_custom_model.py
│── evaluate_models.py
│── compare_models.py
│── run_demo.py
│── requirements.txt
│── README.md

python -m venv venv
source venv/bin/activate       # Windows → venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

data/
├─ train/
│  ├─ images/
│  └─ annotations.json
├─ val/
│  ├─ images/
│  └─ annotations.json
├─ test/
│  ├─ images/
│  └─ annotations.json
device: "cuda"
seed: 42

data:
  format: "coco"
  train_dir: "data/train/images"
  val_dir: "data/val/images"
  train_ann: "data/train/annotations.json"
  val_ann: "data/val/annotations.json"
  classes: ["Taj_Mahal", "Monument", "Temple"]

model:
  name: "custom_detector_v1"
  num_classes: 3
  image_size: [640, 640]
  anchor_scales: [32, 64, 128, 256, 512]

train:
  epochs: 50
  batch_size: 8
  lr: 0.0005
  amp: true

eval:
  iou_thresholds: [0.5, 0.75]
  score_threshold: 0.25
