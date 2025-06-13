import os
import cv2
import numpy as np
import shutil

read_file = "./BUS-UCLM-box/Malignant/HUBL_008.txt"
read_pic = "./BUS-UCLM-box/Malignant/HUBL_008.png"

# check if pic exists
if not os.path.exists(read_pic):
    print("pic not exist")
    exit()
img = cv2.imread(read_pic)
# cv2.imshow('img', img)
# cv2.waitKey(0)
# read file's first line
xywh = []
with open(read_file, 'r') as f:
    line = f.readline()
    xywh = line.split(' ')
    print(xywh)

midx = float(xywh[1]) * img.shape[1]
midy = float(xywh[2]) * img.shape[0]
w = float(xywh[3]) * img.shape[1]
h = float(xywh[4]) * img.shape[0]

cv2.rectangle(img, (int(midx - w / 2), int(midy - h / 2)), (int(midx + w / 2), int(midy + h / 2)), (0, 0, 255), 2)
cv2.imshow('img', img)
cv2.waitKey(0)