import numpy as np
import cv2
import os
import sys
from bounding_box import BoundingBox
from utils import normalized_avg

def merge_bounding_boxes(bounding_boxes):
    # Algo to be improved for better performance

    # Sort by size
    sorted_boxes = sorted(bounding_boxes, key=lambda x: x.size, reverse=False)

    # Merge from smallest to largest
    idx = 0
    while idx < len(sorted_boxes):
        box = sorted_boxes.pop(idx)
        for target in sorted_boxes:
            if box.is_inside(target):
                box = None
                break
        if box:
            sorted_boxes.insert(idx, box)
            idx += 1

    return sorted_boxes


def eval_letter_width(bounding_boxes):
    widths = [box.w for box in bounding_boxes]
    return normalized_avg(widths)


def combine_horizontally(bounding_boxes, evaled_letter_width):
    # Sort by position
    sorted_boxes = sorted(bounding_boxes, key=lambda x: x.x, reverse=False)

    idx = 0
    while idx < len(sorted_boxes)-1:
        box = sorted_boxes[idx]
        next_box = sorted_boxes[idx+1]
        combined = BoundingBox.combine(box, next_box)

        combined_max_width = evaled_letter_width * 1.2

        if combined.w < combined_max_width:
            sorted_boxes.pop(idx)
            sorted_boxes.pop(idx)
            sorted_boxes.insert(idx, combined)
        else:
            idx += 1

    return sorted_boxes


def split_widths(bounding_boxes, evaled_avg_letter_width):
    idx = 0
    splitted = []
    while idx < len(bounding_boxes) - 1:
        box = bounding_boxes[idx]
        width = box.w
        n = round(width / evaled_avg_letter_width)
        if n > 1:
            new_width = int(round(width / n))
            bounding_boxes.pop(idx)
            splitted += [BoundingBox((box.x + incr, 0, new_width, box.h)) for incr in range(0, int(n * new_width), int(new_width))]
        else:
            idx += 1

    return sorted(bounding_boxes + splitted, key=lambda x: x.x)


def get_bounding_boxes(bounding_box_vals, w, h):
    # bounding_boxes = [BoundingBox(bounding_box_val) for bounding_box_val in bounding_box_vals]
    bounding_boxes = [BoundingBox((bounding_box_val[0], 0, bounding_box_val[2], h)) for bounding_box_val in bounding_box_vals]

    # Merge bounding boxes
    bounding_boxes = merge_bounding_boxes(bounding_boxes)

    # Evaluate letter width
    evaled_letter_width, evaled_avg_letter_width = eval_letter_width(bounding_boxes)

    # Combine bounding boxes that intersect but combined width is around letter width
    bounding_boxes = combine_horizontally(bounding_boxes, evaled_letter_width)

    # split boxes of n times width of most boxes into n boxes
    bounding_boxes = split_widths(bounding_boxes, evaled_avg_letter_width)

    return bounding_boxes


def add_border(image):
    bordersize = 1
    border=cv2.copyMakeBorder(image, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])

    return border


def cut_letters(image):
    image = add_border(image)
    output_image = np.array(image, copy=True)

    # Convert Color to white and black
    imgary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Add contrast
    _, thresh = cv2.threshold(imgary, 0, 255, cv2.THRESH_OTSU)
    # _, thresh = cv2.threshold(imgary, 127, 255, cv2.THRESH_BINARY)

    # Find contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Get bounding boxes
    bounding_box_vals = [cv2.boundingRect(contour) for contour in contours]

    # Remove the one with same size as image
    for idx, val in enumerate(bounding_box_vals):
        if val[0] == 0 and val[1] == 0 and val[2] == image.shape[1] and val[3] == image.shape[0]:
            bounding_box_vals.pop(idx)
            break

    # Merge bounding boxes and write to output_image, crop and save images.
    bounding_boxes = get_bounding_boxes(bounding_box_vals, image.shape[1], image.shape[0])
    letters = []
    boxes = []
    for bounding_box in bounding_boxes:
        x, y, w, h = bounding_box.x, bounding_box.y, bounding_box.w, bounding_box.h
        letters.append(output_image[y: y + h, x: x + w])
        boxes.append((x, y, w, h))

    return letters, boxes
