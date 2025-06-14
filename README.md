# MedAI_final_project

## Correcting Box
The file is for the doctor to correct the incorrect boxes.
This was aimed to have a yolo model to have a better and closer predictions. However, when we later fintuned the model using the corrected data, we got a lower accuracy, therefore we did not write this part on the report.

## Mask to Box
### qambi-to-box.py
Run to create Yolo Box from the qamebi masks! :D
* Dataset Format
```
qamebi/
├── Benign/
│   ├── 1 Benign Image.bmp
│   ├── 1 Benign Mask.tif
│   ├── 2 Benign Image.bmp
│   ├── 2 Benign Mask.tif
│   └── ...
├── Malignant/
│   ├── 1 Malignant Image.bmp
│   ├── 1 Malignant Mask.tif
│   ├── 2 Malignant Image.bmp
│   ├── 2 Malignant Mask.tif
│   └── ...
```

### uclm-to-box.py
Run to create Yolo Box from the BUS-UCLM masks! :D
* Dataset Format
```
BUS-UCLM/
├── images/
│   ├── ALWI_000.png
│   ├── ALWI_001.png
│   └── ...
├── masks/
│   ├── ALWI_000.png
│   ├── ALWI_001.png
│   └── ...
├── INFO.csv
```

### show.py
A simple python script for you to make sure if you have the correct box.

## YOLO Training
* Dataset Format
```
BUSI-YOLOv8/
├── images/
│   └── train/
│       ├── img1.jpg
│       ├── img2.jpg
│       └── ...
│   └── val/
│       ├── img1.jpg
│       ├── img2.jpg
│       └── ...
├── labels/
│    └── train/
│        ├── img1.txt
│        ├── img2.txt
│        └── ...
│    └── val/
│        ├── img1.txt
│        ├── img2.txt
│        └── ...
```

* Data Augmentation
  - use augment.py to augment data : **apply_aug_only.py** 
  - use augment.py +  clahe.py to augment data  : **apply_augmentations.py**
  - use clahe.py to augment data : **apply_clahe.py** 
  - visualize augment.py + clahe.py results : **visualize_augmented_data.py**
  - data augmentation code : **clahe.py** & **augment.py**
 
* YOLO Training
  - Modify **data.yaml** for different yolo training data
  - Training command : 
    ```yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=256 batch=16 device="cpu"```

## classify_with_crop
### YOLO Dataset Preprocessing

**Script:** `yolo_dataset_preprocess.ipynb`

### Tasks:
- Load image–box pairs from `benign_yolo/` and `malignant_yolo/`
- Crop bounding boxes
- Pad to square, resize to 224×224
- Convert to grayscale
- Save final image under `processed/benign/` or `processed/malignant/`

**Output Folder Example:**
```
QAMEBI_CLAHE/processed/
├── benign/
│ ├── benign (1).png
│ ├── benign (2).png
│ └── ...
├── malignant/
│ ├── malignant (1).png
│ ├── malignant (2).png
│ └── ...
```
---

## VGG16 Classification

**Script:** `pretrained_VGG16_on_cropped.ipynb`

### Model Details:
- Base: Pretrained `torchvision.models.vgg16`
- Head: Modified to output 1 sigmoid unit for binary classification
- Training: 
  - BCE Loss
  - Adam optimizer
  - 10 epochs

### Data Pipeline:
- `train/val/test` split: 70/15/15
- Grayscale → 3-channel (for VGG) → Resize (224×224) → Normalize

### Metrics:
- Accuracy
- Precision, Recall (Sensitivity), Specificity
- F1 Score, AUROC
- Confusion Matrix
- Bootstrapped Metric Distributions

---

## Calibration & Reliability

- Applied **Platt Scaling** on validation set
- Plotted **Calibration Curve**
- Evaluated **Brier Score** for probabilistic reliability

---

## Explainability (Grad-CAM)

Used Grad-CAM to visualize which image regions influenced the classifier.

- Applied only to confident malignant predictions
- Heatmaps generated and overlaid on grayscale images

---

## Requirements

(For `classify_with_crop/`)
- Python 3.8+
- PyTorch
- OpenCV
- `torchvision`, `matplotlib`, `seaborn`
- `pytorch-grad-cam`

---

## 📌 How to Run

1. Generate YOLO boxes from mask:
    ```bash
    python qambi-to-box.py
    ```

2. Run preprocessing notebook:
    - `yolo_dataset_preprocess.ipynb`

3. Train classifier:
    - `pretrained_VGG16_on_cropped.ipynb`

---

## 📧 Contact

Feel free to reach out for questions or collaboration ideas!
