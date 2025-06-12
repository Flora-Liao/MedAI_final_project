# MedAI_final_project

# YOLO Training
* Dataset Format
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

* Data Augmentation
  - use augment.py to augment data : **apply_aug_only.py** 
  - use augment.py +  clahe.py to augment data  : **apply_augmentations.py**
  - use clahe.py to augment data : **apply_clahe.py** 
  - visualize augment.py + clahe.py results : **visualize_augmented_data.py**
  - data augmentation code : **clahe.py** & **augment.py**
 
* YOLO Training
  - Modify **data.yaml** for different yolo training data
  - Training command
    ```yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=256 batch=16 device="cpu"```
