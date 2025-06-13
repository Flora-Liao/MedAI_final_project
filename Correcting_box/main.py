import cv2
import os
import math

# Global variables
drawing = False
ix, iy = -1, -1
old_boxes_for_each_picture = {}
new_boxes_for_each_picture = {}
cur_boxes = []
app_state = 0

pic_path = "./pic/"
save_file = "./new_labels"
label_path = "./labels/"

# for every file in label_path
for file in os.listdir(label_path):
    # if the file not text
    if not file.endswith(".txt"):
        continue

    # open the file
    with open(os.path.join(label_path, file), "r") as f:
        # key name is the file name
        file = file.split("_")[-1]
        key = file[:-4] + ".png"
        # add the boxes to the dictionary
        temp = []
        for line in f.readlines():
            line = line.split(" ")
            temp.append([float(line[1]), float(line[2]), float(line[3]), float(line[4])])
        old_boxes_for_each_picture[key] = temp

print(str(len(old_boxes_for_each_picture)) + " labels found")
# for every file in the folder
for file in os.listdir(pic_path):
    # if the file not image
    if not file.endswith(".png"):
        continue

    # open the image
    img = cv2.imread(os.path.join(pic_path, file))
    img_copy = img.copy()
    app_state = 0
    
    # create a window
    cv2.namedWindow("View")
    print(file)

    # find the boxes in old_boxes
    if file in old_boxes_for_each_picture:
        for box in old_boxes_for_each_picture[file]:
            midx = int(box[0] * img.shape[1])
            midy = int(box[1] * img.shape[0])
            width = int(box[2] * img.shape[1])
            height = int(box[3] * img.shape[0])
            minx = midx - width // 2
            miny = midy - height // 2
            maxx = midx + width // 2
            maxy = midy + height // 2
            
            cv2.rectangle(img_copy, (minx, miny), (maxx, maxy), (0, 0, 255), 2)
            
    # show the image
    cv2.imshow("View", img_copy)
    
    while True:
        key = cv2.waitKey(0)
        # if key is a
        if key == ord('a'):
            app_state = 1
            break
        # if key is s
        elif key == ord('s'):
            break

    cv2.destroyWindow("View")
    # if state still 0, continue
    if app_state == 0:
        continue

    # create edit window
    img_copy = img.copy()
    cur_boxes = []
    cv2.namedWindow("Edit")
    cv2.imshow("Edit", img_copy)

    def get_cords_from_lines(array, img):
        minx, miny, maxx, maxy = img.shape[1], img.shape[0], 0, 0
        if len(array) > 1:
            for line in array:
                minx = min(minx, min(line[0][0], line[1][0]))
                miny = min(miny, min(line[0][1], line[1][1]))
                maxx = max(maxx, max(line[0][0], line[1][0]))
                maxy = max(maxy, max(line[0][1], line[1][1]))
        else:
            max_length = max(abs(array[0][0][0] - array[0][1][0]), abs(array[0][0][1] - array[0][1][1]))
            middle_point = ((array[0][0][0] + array[0][1][0]) // 2, (array[0][0][1] + array[0][1][1]) // 2)
            
            minx = max(middle_point[0] - max_length // 2, 0)
            miny = max(middle_point[1] - max_length // 2, 0)
            maxx = min(middle_point[0] + max_length // 2, img.shape[1])
            maxy = min(middle_point[1] + max_length // 2, img.shape[0])

        x_length = maxx - minx
        y_length = maxy - miny
        minx = max(minx - math.sqrt(x_length), 0)
        miny = max(miny - math.sqrt(y_length), 0)
        maxx = min(maxx + math.sqrt(x_length), img.shape[1])
        maxy = min(maxy + math.sqrt(y_length), img.shape[0])

        return int(minx), int(miny), int(maxx), int(maxy)

    def drawing_boxes():
        global cur_boxes, img_copy, img

        # copy the image
        img_copy = img.copy()
        cnt = 0
        # boxes part
        while cnt < len(cur_boxes):
            if len(cur_boxes[cnt]) == 2:
                break
            cv2.rectangle(img_copy, (cur_boxes[cnt][0], cur_boxes[cnt][1]), (cur_boxes[cnt][2], cur_boxes[cnt][3]), (0, 255, 0), 2)
            cnt += 1

        # line part
        if cnt < len(cur_boxes):
            minx, miny, maxx, maxy = get_cords_from_lines(cur_boxes[cnt:], img_copy)
            while cnt < len(cur_boxes):
                cv2.line(img_copy, cur_boxes[cnt][0], cur_boxes[cnt][1], (0, 255, 0), 2)
                cnt += 1
            cv2.rectangle(img_copy, (minx, miny), (maxx, maxy), (255, 0, 0), 2)
        
        cv2.imshow("Edit", img_copy)

    # function clicky
    def clicky(event, x, y, flags, param):
        global ix, iy, drawing, cur_boxes, app_state
        if event == cv2.EVENT_LBUTTONDOWN:
            if drawing == False:
                drawing = True
                ix, iy = x, y
            else:
                if app_state == 1:
                    cur_boxes.append([(ix, iy), (x, y)])
                elif app_state == 2:
                    cur_boxes.append([ix, iy, x, y])

                drawing = False
                drawing_boxes()
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                drawing_boxes()
                if app_state == 1:
                    cv2.line(img_copy, (ix, iy), (x, y), (255, 0, 0), 2)
                elif app_state == 2:
                    cv2.rectangle(img_copy, (ix, iy), (x, y), (255, 0, 0), 2)
                cv2.imshow("Edit", img_copy)

    # set the clicky function
    cv2.setMouseCallback("Edit", clicky)

    while True:
        key = cv2.waitKey(0)
        # if key is a
        if key == ord('a'):
            # pop last element
            if len(cur_boxes) > 0:
                cur_boxes.pop()
            drawing_boxes()
        # if key is s
        elif key == ord('s'):
            real_save = True
            # check if the last element is a line
            if len(cur_boxes) > 0 and len(cur_boxes[-1]) == 2:
                real_save = False

                # turn all the lines to a box
                first_line = 0
                for box in cur_boxes:
                    if len(box) == 2:
                        break
                    first_line += 1

                minx, miny, maxx, maxy = get_cords_from_lines(cur_boxes[first_line:], img_copy)
                cur_boxes = cur_boxes[:first_line]
                cur_boxes.append([minx, miny, maxx, maxy])
                drawing_boxes()
            if real_save:
                normalized = []
                for box in cur_boxes:
                    midx = (box[0] + box[2]) // 2
                    midy = (box[1] + box[3]) // 2
                    width = box[2] - box[0]
                    height = box[3] - box[1]
                    normalized.append([midx / img.shape[1], midy / img.shape[0], width / img.shape[1], height / img.shape[0]])
                new_boxes_for_each_picture[file] = normalized
                break
        elif key == ord('q'):
            if app_state == 1:
                # pop all the lines
                while len(cur_boxes) > 0 and len(cur_boxes[-1]) == 2:
                    cur_boxes.pop()
                drawing_boxes()

                app_state = 2
            else:
                app_state = 1

    cv2.destroyWindow("Edit")

if not os.path.exists(save_file):
    os.makedirs(save_file)

for file in new_boxes_for_each_picture:
    txt = file[:-4] + ".txt"
    with open(os.path.join(save_file, txt), "w") as f:
        for box in new_boxes_for_each_picture[file]:
            f.write("0 " + str(box[0]) + " " + str(box[1]) + " " + str(box[2]) + " " + str(box[3]) + "\n")