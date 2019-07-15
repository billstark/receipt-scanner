import cv2
import numpy as np
import os
from bounding_box import BoundingBox
from utils import normalized_avg

def eval_line_height(bounding_boxes):
    heights = [box.h for box in bounding_boxes]
    return normalized_avg(heights)


def split_heights(bounding_boxes, evaled_avg_line_height):
    idx = 0
    splitted = []
    while idx < len(bounding_boxes) - 1:
        box = bounding_boxes[idx]
        height = box.h
        n = int(height / evaled_avg_line_height)
        if n > 1:
            new_height = int(round(height / n))
            bounding_boxes.pop(idx)
            splitted += [BoundingBox((box.x, box.y + incr, box.w, new_height)) for incr in range(0, int(new_height * n), new_height)]
        else:
            idx += 1

    return sorted(bounding_boxes + splitted, key=lambda x: x.y)


def seperate_n_lines(bounding_boxes):
    # Evaluate letter width
    evaled_avg_line_height, _ = eval_line_height(bounding_boxes)

    # split boxes of n times width of most boxes into n boxes
    bounding_boxes = split_heights(bounding_boxes, evaled_avg_line_height)

    return bounding_boxes


def cut_lines(image, for_crnn=True):
    # Copy for labelling
    labelled = image.copy()

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binary
    _, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)

    # Dilation
    kernel = np.ones((3, 15 if for_crnn else 100), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    # Find, sort contours
    ctrs, _ = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[1])

    # Cut out lines
    bounding_boxes = []
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        bounding_boxes.append(BoundingBox((x, y, w, h)))

    height, width, _ = image.shape
    bounding_boxes = seperate_n_lines(bounding_boxes)
    lines = []
    boxes = []
    for bounding_box in bounding_boxes:
        x, y, w, h = bounding_box.x, bounding_box.y, bounding_box.w, bounding_box.h
        size_dilate_val = int(h / 7)
        ny = max(y - size_dilate_val, 0)
        ny2 = min(y + h + size_dilate_val, height)
        nx = max(x - size_dilate_val, 0)
        nx2 = min(x + w + size_dilate_val, width)
        line = image[ny:ny2, nx:nx2]
        lines.append(line)
        cv2.rectangle(labelled, (nx, ny), (nx2, ny2), (90, 0, 255), 2)
        boxes.append((nx, ny, nx2-nx, ny2-ny))

    return labelled, lines, boxes

# labelled, _, _ =cut_lines(cv2.imread('test.png'))
# cv2.imshow('l', labelled)
# cv2.waitKey(0)
