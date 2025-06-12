import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from clahe import extract_clahe_samples
from augment import build_transforms

# 設定資料路徑
input_dir = "BUSI-fewshot/images/train"
label_dir = "BUSI-fewshot/labels/train"
out_img_dir = "BUSI-fewshot2/images/train"
out_lbl_dir = "BUSI-fewshot2/labels/train"

# 建立資料夾
os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_lbl_dir, exist_ok=True)

# 建立資料增強設定
cfg = {
    "rotation": 10,
    "h_flip": True,
    "crop_resize": 0.8,
    "elastic": True,
    "speckle_noise": (0.01, 0.03),
}
tfms = build_transforms(cfg, totensor=False)

# CLAHE operator
clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# 讀取所有影像
image_paths = sorted(glob(os.path.join(input_dir, "*.png")) + glob(os.path.join(input_dir, "*.jpg")))

for img_path in tqdm(image_paths):
    base = os.path.basename(img_path)
    name_no_ext = os.path.splitext(base)[0]
    label_path = os.path.join(label_dir, base.replace(".jpg", ".txt").replace(".png", ".txt"))

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    clahe_img = clahe_op.apply(img.copy())

    h, w = img.shape[:2]

    # --- 讀取原始標註 ---
    if not os.path.exists(label_path):
        continue
    with open(label_path, "r") as f:
        lines = f.readlines()
    bboxes, class_ids = [], []
    for line in lines:
        cid, x, y, bw, bh = map(float, line.strip().split())
        x1 = (x - bw / 2) * w
        y1 = (y - bh / 2) * h
        x2 = (x + bw / 2) * w
        y2 = (y + bh / 2) * h
        bboxes.append([x1, y1, x2, y2])
        class_ids.append(int(cid))

    # === 儲存原始圖與標註 ===
    cv2.imwrite(os.path.join(out_img_dir, f"{name_no_ext}_original.jpg"), img)
    with open(os.path.join(out_lbl_dir, f"{name_no_ext}_original.txt"), "w") as f:
        for cid, bbox in zip(class_ids, bboxes):
            x1, y1, x2, y2 = bbox
            x_center = (x1 + x2) / 2 / w
            y_center = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f"{cid} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

    # === 儲存 CLAHE 圖與標註（框一樣） ===
    cv2.imwrite(os.path.join(out_img_dir, f"{name_no_ext}_clahe.jpg"), clahe_img)
    with open(os.path.join(out_lbl_dir, f"{name_no_ext}_clahe.txt"), "w") as f:
        for cid, bbox in zip(class_ids, bboxes):
            x1, y1, x2, y2 = bbox
            x_center = (x1 + x2) / 2 / w
            y_center = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f"{cid} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

    # === CLAHE + Albumentations 增強後圖與標註 ===
    augmented = tfms(image=clahe_img, bboxes=bboxes, class_labels=class_ids)
    aug_img = augmented["image"]
    aug_bboxes = augmented["bboxes"]
    aug_labels = augmented["class_labels"]

    cv2.imwrite(os.path.join(out_img_dir, f"{name_no_ext}_aug.jpg"), aug_img)
    with open(os.path.join(out_lbl_dir, f"{name_no_ext}_aug.txt"), "w") as f:
        for cid, bbox in zip(aug_labels, aug_bboxes):
            x1, y1, x2, y2 = bbox
            x_center = (x1 + x2) / 2 / w
            y_center = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f"{cid} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")
