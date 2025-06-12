import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

# 資料路徑設定
input_dir = r"C:\Users\User\Desktop\MedAI\project\Yolo_train\BUS-UCLM-box\Malignant"
label_dir = r"C:\Users\User\Desktop\MedAI\project\Yolo_train\BUS-UCLM-box\Malignant"
out_img_dir = "bus-malignant-clahe/images/train"
out_lbl_dir = "bus-malignant-clahe/labels/train"

# 建立輸出資料夾
os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_lbl_dir, exist_ok=True)

# CLAHE operator
clahe_op = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

# 載入所有影像
image_paths = sorted(glob(os.path.join(input_dir, "*.png")) + glob(os.path.join(input_dir, "*.jpg")))

for img_path in tqdm(image_paths):
    base = os.path.basename(img_path)
    name_no_ext = os.path.splitext(base)[0]
    label_path = os.path.join(label_dir, base.replace(".jpg", ".txt").replace(".png", ".txt"))

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    clahe_img = clahe_op.apply(img.copy())
    h, w = img.shape[:2]

    # 若無對應標註則略過
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
    # cv2.imwrite(os.path.join(out_img_dir, f"{name_no_ext}_original.jpg"), img)
    # with open(os.path.join(out_lbl_dir, f"{name_no_ext}_original.txt"), "w") as f:
    #     for cid, bbox in zip(class_ids, bboxes):
    #         x1, y1, x2, y2 = bbox
    #         x_center = (x1 + x2) / 2 / w
    #         y_center = (y1 + y2) / 2 / h
    #         bw = (x2 - x1) / w
    #         bh = (y2 - y1) / h
    #         f.write(f"{cid} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

    # === 儲存 CLAHE 圖與標註（標註相同） ===
    cv2.imwrite(os.path.join(out_img_dir, f"{name_no_ext}_clahe.jpg"), clahe_img)
    with open(os.path.join(out_lbl_dir, f"{name_no_ext}_clahe.txt"), "w") as f:
        for cid, bbox in zip(class_ids, bboxes):
            x1, y1, x2, y2 = bbox
            x_center = (x1 + x2) / 2 / w
            y_center = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f"{cid} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")
