import cv2
import numpy as np
import os
from bounding_box import BoundingBox

def eval_line_height(bounding_boxes):
    heights = [box.h for box in bounding_boxes]
    average = np.average(heights)
    variance = np.var(heights)
    if variance == 0:
        return heights[0], heights[0]
    n = len(heights)
    tolerance = 4 # Higher for more tolerance over outliers
    filtered_heights = []
    while True:
        filtered_heights = [height for height in heights if -tolerance <= (height - average)/np.sqrt(variance/n) <= tolerance]
        if not filtered_heights:
            tolerance *= 1.5
        else:
            break

    return (np.max(filtered_heights), np.average(filtered_heights))


def split_heights(bounding_boxes, evaled_avg_line_height):
    idx = 0
    splitted = []
    while idx < len(bounding_boxes) - 1:
        box = bounding_boxes[idx]
        height = box.h
        n = round(height / evaled_avg_line_height)
        if n > 1:
            new_height = int(round(height / n))
            bounding_boxes.pop(idx)
            splitted += [BoundingBox((box.x, box.y + incr, box.w, new_height)) for incr in range(0, int(new_height * n), new_height)]
        else:
            idx += 1

    return sorted(bounding_boxes + splitted, key=lambda x: x.y)


def seperate_n_lines(bounding_boxes):
    # Evaluate letter width
    evaled_line_height, evaled_avg_line_height = eval_line_height(bounding_boxes)

    # split boxes of n times width of most boxes into n boxes
    bounding_boxes = split_heights(bounding_boxes, evaled_avg_line_height)

    return bounding_boxes


def cut_lines(image):
    # Copy for labelling
    labelled = image.copy()

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binary
    ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)

    # Dilation
    kernel = np.ones((3, 100), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    # Find, sort contours
    im2,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[1])

    # Cut out lines
    bounding_boxes = []
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        bounding_boxes.append(BoundingBox((x, y, w, h)))

    bounding_boxes = seperate_n_lines(bounding_boxes)
    lines = []
    boxes = []
    for bounding_box in bounding_boxes:
        x, y, w, h = bounding_box.x, bounding_box.y, bounding_box.w, bounding_box.h
        line = image[y:y+h, x:x+w]
        lines.append(line)
        cv2.rectangle(labelled,(x,y),( x + w, y + h ),(90,0,255),2)
        boxes.append((x, y, w, h))

    return labelled, lines, boxes
