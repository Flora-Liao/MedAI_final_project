import os
import cv2
import numpy as np
import shutil

input_folder = './qamebi'
output_folder = './qamebi-box'

benign = '/Benign'
mal = '/Malignant'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if not os.path.exists(output_folder + benign):
    os.makedirs(output_folder + benign)

if not os.path.exists(output_folder + mal):
    os.makedirs(output_folder + mal)

# for benign
def func(file_dir, output_dir):
    for file in os.listdir(input_folder + file_dir):
        # find file ends with Image.bmp
        if file.endswith('Image.bmp'):
            # get file number, numbers in filename
            file_number = ''.join([c for c in file if c.isdigit()])
            # copy file
            img = cv2.imread(input_folder + file_dir + '/' + file)
            shutil.copyfile(input_folder + file_dir + '/' + file, output_dir + '/' + file_dir[1:] + str(file_number) + '.png')

            # find mask by "Image.bmp" to "Mask.tif"
            mask = file.replace('Image.bmp', 'Mask.tif')
            # read mask
            mask = cv2.imread(input_folder + file_dir + '/' + mask)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            # search for minx, miny, maxx, maxy
            minx = 1e9
            miny = 1e9
            maxx = 0
            maxy = 0
            for y in range(mask.shape[0]):
                for x in range(mask.shape[1]):
                    if mask[y][x] > 127:
                        minx = min(minx, x)
                        miny = min(miny, y)
                        maxx = max(maxx, x)
                        maxy = max(maxy, y)

            w = maxx - minx
            h = maxy - miny
            midx = minx + w / 2
            midy = miny + h / 2
            
            w += np.sqrt(w)
            h += np.sqrt(h)

            midx /= img.shape[1]
            midy /= img.shape[0]
            w /= img.shape[1]
            h /= img.shape[0]

            # write to output_dir + '/' + file_dir[1:] + str(file_number) + '.txt'
            with open(output_dir + '/' + file_dir[1:] + str(file_number) + '.txt', 'w') as f:
                f.write('0 ' + str(midx) + ' ' + str(midy) + ' ' + str(w) + ' ' + str(h))


func(benign, output_folder + benign)
func(mal, output_folder + mal)