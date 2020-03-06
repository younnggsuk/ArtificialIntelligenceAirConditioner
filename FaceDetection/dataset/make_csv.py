import os
import numpy as np
import cv2

CUR_PATH = os.getcwd()

TRAIN_BOX_LIST_PATH = os.path.join(CUR_PATH, "wider_face_split")
TRAIN_BOX_LIST_PATH = os.path.join(
    TRAIN_BOX_LIST_PATH, "wider_face_train_bbx_gt.txt")

TRAIN_CSV_PATH = os.path.join(
    CUR_PATH, "train.csv")
TRAIN_IMG_PATH = os.path.join(CUR_PATH, "WIDER_train")


VAL_BOX_LIST_PATH = os.path.join(CUR_PATH, "wider_face_split")
VAL_BOX_LIST_PATH = os.path.join(
    VAL_BOX_LIST_PATH, "wider_face_val_bbx_gt.txt")

VAL_CSV_PATH = os.path.join(
    CUR_PATH, "val.csv")
VAL_IMG_PATH = os.path.join(CUR_PATH, "WIDER_val")

# Class
CLASS = "face"

# Make Train csv
with open(TRAIN_CSV_PATH, "w") as target:
    target.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")
    with open(TRAIN_BOX_LIST_PATH, "r") as f:
        while(True):
            file_path = f.readline()

            # File end
            if not file_path:
                break

            # File Name
            file_name = file_path[file_path.find('/')+1:-1]

            # Image's width, height
            img_path = os.path.join(TRAIN_IMG_PATH, file_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            height, width, _ = img.shape

            # Number of Boxes
            num_box = f.readline()
            num_box = int(num_box)

            # Print & Delete Non-box image
            if (num_box == 0):
                print(img_path)
                os.remove(img_path)
                file_path = f.readline()
                continue

            # Coordinates of Box
            for i in range(num_box):
                box = f.readline()
                start_idx = 0
                # coord = [x1, y1, width, height]
                coords = []

                for j in range(4):
                    next_idx = box.find(' ')
                    coords.append(int(box[start_idx:next_idx]))
                    box = box[next_idx+1:]
                    next_idx = start_idx

                # coord = [x1, y1, x2, y2]
                coords[2] += coords[0]
                coords[3] += coords[1]

                data = "%s,%s,%s,%s,%s,%s,%s,%s\n" % (
                    file_name, width, height, CLASS, coords[0], coords[1], coords[2], coords[3])
                target.write(data)


# Make Val csv
with open(VAL_CSV_PATH, "w") as target:
    target.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")
    with open(VAL_BOX_LIST_PATH, "r") as f:
        while(True):
            file_path = f.readline()

            # File end
            if not file_path:
                break

            # File Name
            file_name = file_path[file_path.find('/')+1:-1]

            # Image's width, height
            img_path = os.path.join(VAL_IMG_PATH, file_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            height, width, _ = img.shape

            # Number of Boxes
            num_box = f.readline()
            num_box = int(num_box)

            # Print & Delete Non-box image
            if (num_box == 0):
                print(img_path)
                os.remove(img_path)
                file_path = f.readline()
                continue

            # Coordinates of Box
            for i in range(num_box):
                box = f.readline()
                start_idx = 0
                # coord = [x1, y1, width, height]
                coords = []

                for j in range(4):
                    next_idx = box.find(' ')
                    coords.append(int(box[start_idx:next_idx]))
                    box = box[next_idx+1:]
                    next_idx = start_idx

                # coord = [x1, y1, x2, y2]
                coords[2] += coords[0]
                coords[3] += coords[1]

                data = "%s,%s,%s,%s,%s,%s,%s,%s\n" % (
                    file_name, width, height, CLASS, coords[0], coords[1], coords[2], coords[3])
                target.write(data)
