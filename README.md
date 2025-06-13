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
