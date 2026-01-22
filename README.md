

## 🚀 Anonymous Running Instructions

This repository contains the code and experimental setup for our 2026 work on HMPE-Bone. Below are the standard procedures to run the project and reproduce our results.

### 📊 Evaluation Results

| Method | Date | Backbone | Epoch | GFLOPs | Parameters (MB) | PASCAL VOC mAP | PASCAL VOC mAP@0.95 | NWPU mAP | NWPU mAP@0.95 |
|--------|------|----------|-------|--------|-----------------|----------------|---------------------|----------|---------------|
| Our work | 2026 | HMPE-Bone | 100 | 57 | 66.3 | 70.62 | 51.59 | 94.51 | 67.20 |

---

### 📁 Project Structure Overview

```
.
├── dataset/                 # Dataset directory
│   ├── images/              # Image folders (train, val, test, etc.)
│   └── labels/              # Label folders (YOLO format)
├── weights/                 # Model weights save path
├── runs/                    # Training/detection logs and output results
├── ultralytics/             # Ultralytics framework source code (models, engine, nn modules, etc.)
└── result/                  # Custom result output directory
```

---

### ⚙️ Environment Setup

Recommended: Python >= 3.8. Install dependencies with:

```bash
pip install -r requirements.txt
```

For advanced features (DCNv4, Mamba, etc.), please refer to instructions in `ultralytics/nn/extra_modules/` for additional compilation or installation.

---

### 🧪 Model Training

To train the RT-DETR model, run the following command:

```bash
python train.py \
    --model rt-detr-l.yaml \
    --data your_dataset.yaml \
    --epochs 100 \
    --batch-size 16 \
    --imgsz 640 \
    --device 0 \
    --project runs/train \
    --name exp_anonymous
```

Parameters explanation:
- `--model`: Model configuration file path (located in `ultralytics/cfg/models/rt-detr/`)
- `--data`: Dataset configuration file path (recommended with absolute path)
- `--project`: Main directory for experiment results
- `--name`: Current experiment name

---

### 🔍 Model Validation / Inference

#### Single Image Inference

```bash
python detect.py \
    --weights weights/best.pt \
    --source images/sample.jpg \
    --imgsz 640 \
    --conf-thres 0.25 \
    --iou-thres 0.45 \
    --device 0 \
    --save-txt \
    --project runs/detect \
    --name exp_anonymous_detect
```

#### Validation on Validation Set

```bash
python val.py \
    --weights weights/best.pt \
    --data your_dataset.yaml \
    --imgsz 640 \
    --batch-size 16 \
    --conf-thres 0.001 \
    --iou-thres 0.7 \
    --device 0 \
    --project runs/val \
    --name exp_anonymous_val
```

---

### 📊 Training and Validation Visualization

Metrics are automatically logged to `runs/train/exp_anonymous/` during training. View with TensorBoard:

```bash
tensorboard --logdir runs/train/exp_anonymous/
```

---

### 🧹 Clean Cache (Optional)

Remove Python cache files:

```bash
find . -type d -name __pycache__ | xargs rm -rf
```

---

All outputs will be saved in the `runs/` folder by default for easy management and experiment reproduction.

---

> ⚠️ Note: This is an anonymous shared version. Sensitive paths and configurations have been sanitized. Please modify relevant paths according to your local environment before use.

---

