import os
import cv2
import numpy as np
import shutil

input_folder = './BUS-UCLM'
input_images = '/images'
input_masks = '/masks'
output_folder = './BUS-UCLM-box'

benign = '/Benign'
mal = '/Malignant'
normal = '/Normal'
others = '/Others'

info = '/INFO.csv'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if not os.path.exists(output_folder + benign):
    os.makedirs(output_folder + benign)
if not os.path.exists(output_folder + mal):
    os.makedirs(output_folder + mal)
if not os.path.exists(output_folder + normal):
    os.makedirs(output_folder + normal)
if not os.path.exists(output_folder + others):
    os.makedirs(output_folder + others)

def get_boxes(file):
    ret = []
    # convert to grayscale
    print(file)
    file = cv2.imread(file)
    file = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
    for y in range(file.shape[0]):
        for x in range(file.shape[1]):
            if file[y][x] > 0:
                cur_points = [[y, x]]
                minx = x
                miny = y
                maxx = x
                maxy = y
                while len(cur_points) > 0:
                    cur_point = cur_points.pop(0)
                    if cur_point[0] < 0 or cur_point[1] < 0 or cur_point[0] >= file.shape[0] or cur_point[1] >= file.shape[1]:
                        continue
                    if file[cur_point[0]][cur_point[1]] == 0:
                        continue
                    file[cur_point[0]][cur_point[1]] = 0
                    minx = min(minx, cur_point[1])
                    miny = min(miny, cur_point[0])
                    maxx = max(maxx, cur_point[1])
                    maxy = max(maxy, cur_point[0])
                    cur_points.append([cur_point[0] - 1, cur_point[1]])
                    cur_points.append([cur_point[0] + 1, cur_point[1]])
                    cur_points.append([cur_point[0], cur_point[1] - 1])
                    cur_points.append([cur_point[0], cur_point[1] + 1])

                midx = (minx + maxx) / 2
                midy = (miny + maxy) / 2
                w = maxx - minx
                h = maxy - miny
                w += np.sqrt(w)
                h += np.sqrt(h)
                ret.append([midx / file.shape[1], midy / file.shape[0], w / file.shape[1], h / file.shape[0]])

    
    return ret

# for lines in info, theres header
for line in open(input_folder + info, encoding='utf-8').readlines()[1:]:
    # decode as utf-8
    #line = line.decode("utf-8")
    line = line.split(',')
    filename = line[0]
    label = line[2]
    
    boxes = get_boxes(input_folder + input_masks + '/' + filename)

    destination = './'
    
    if line[5] == 'Yes\n':
        destination = output_folder + others
    elif label == 'Benign':
        destination = output_folder + benign
    elif label == 'Malignant':
        destination = output_folder + mal
    elif label == 'Normal':
        destination = output_folder + normal
    else:
        print('unknown label')
    
    shutil.copyfile(input_folder + input_images + '/' + filename, destination + '/' + filename)
    with open(destination + '/' + filename[:-4] + '.txt', 'w') as f:
        for box in boxes:
            f.write('0 ' + str(box[0]) + ' ' + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + '\n')
    