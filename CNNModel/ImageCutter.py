import numpy as np
import cv2
import os
import sys

IMG_PATH = './trainingImages'
CROP_RESULT_PATH = 'cropedResults'

def training_img_path(img_name):
    return os.path.join(IMG_PATH, img_name)


def cropped_result_path(img_name):
    return os.path.join(IMG_PATH, CROP_RESULT_PATH, img_name)


def merge_bounding_boxes(bounding_boxes):
    # Algo to be improved for better performance

    # Sort by size
    sorted_boxes = sorted(bounding_boxes, key=lambda x: x.size, reverse=False)

    # Merge from smallest to largest
    idx = 0
    while idx < len(sorted_boxes):
        box = sorted_boxes.pop(idx)
        for t_idx, target in enumerate(sorted_boxes):
            if box.is_inside(target):
                box = None
                break
        if box:
            sorted_boxes.insert(idx, box)
            idx += 1

    return sorted_boxes


def eval_letter_width(bounding_boxes):
    widths = [box.w for box in bounding_boxes]
    average = np.average(widths)
    variance = np.var(widths)
    n = len(widths)
    tolerance = 4 # Higher for more tolerance over outliers
    filtered_widths = [width for width in widths if -tolerance <= (width - average)/np.sqrt(variance/n) <= tolerance]
    return np.max(filtered_widths)


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


def get_bounding_boxes(bounding_box_vals, w, h):
    # bounding_boxes = [BoundingBox(bounding_box_val) for bounding_box_val in bounding_box_vals]
    bounding_boxes = [BoundingBox((bounding_box_val[0], 0, bounding_box_val[2], h)) for bounding_box_val in bounding_box_vals]

    # Merge bounding boxes
    bounding_boxes = merge_bounding_boxes(bounding_boxes)

    # Evaluate letter width
    evaled_letter_width = eval_letter_width(bounding_boxes)

    # Combine bounding boxes that intersect but combined width is around letter width
    bounding_boxes = combine_horizontally(bounding_boxes, evaled_letter_width)

    return bounding_boxes


class BoundingBox(object):
    def __init__(self, bounding_box_vals):
        self.x, self.y, self.w, self.h = bounding_box_vals
        self.size = self.w * self.h

    def is_inside(self, other):
        return self.x >= other.x and\
               self.y >= other.y and\
               self.x + self.w <= other.x + other.w and\
               self.y + self.h <= other.y + other.h

    @staticmethod
    def combine(b1, b2):
        x = min(b1.x, b2.x)
        y = min(b1.y, b2.y)
        w = max(b1.x + b1.w, b2.x + b2.w) - x
        h = max(b1.y + b1.h, b2.y + b2.h) - y
        return BoundingBox((x, y, w, h))


def add_border(im):
    row, col= im.shape[:2]
    bottom= im[row-2:row, 0:col]

    bordersize=1
    border=cv2.copyMakeBorder(im, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255, 255, 255])

    return border


def test(im_name):
    # Check if is valid path
    if not os.path.isfile(training_img_path(im_name)):
        raise Exception('Invalid file path.')

    # Read img
    im = cv2.imread(training_img_path(im_name))
    im = add_border(im)
    output_image = np.array(im, copy=True)

    # Convert Color to white and black
    imgary = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Add contrast
    _, thresh = cv2.threshold(imgary, 0, 255, cv2.THRESH_OTSU)
    # _, thresh = cv2.threshold(imgary, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(training_img_path('out_thresh.png'), thresh)

    # Find contour
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contoured img
    image_contoured = cv2.drawContours(output_image, contours, -1, (0, 255, 0), 1)
    cv2.imwrite(training_img_path('out_contour.png'), image_contoured)

    # # Find and draw bounding rects for all contours
    # for index, contour in enumerate(contours):
    #     x, y, w, h = cv2.boundingRect(contour)
    #     cv2.imwrite(cropped_result_path('contour_piece_{}.png'.format(index)), output_image[y: y + h, x: x + w])
    #     output_image = cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # Get bounding boxes
    bounding_box_vals = [cv2.boundingRect(contour) for contour in contours]

    # Remove the one with same size as image
    for idx, val in enumerate(bounding_box_vals):
        if val[0] == 0 and val[1] == 0 and val[2] == im.shape[1] and val[3] == im.shape[0]:
            bounding_box_vals.pop(idx)
            break

    # Merge bounding boxes and write to output_image, crop and save images.
    bounding_boxes = get_bounding_boxes(bounding_box_vals, im.shape[1], im.shape[0])
    for index, bounding_box in enumerate(bounding_boxes):
        x, y, w, h = bounding_box.x, bounding_box.y, bounding_box.w, bounding_box.h
        cv2.imwrite(cropped_result_path('box_piece_{}.png'.format(index)), output_image[y: y + h, x: x + w])
        output_image = cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 1)


    cv2.imwrite(training_img_path('out_boxed.png'), output_image)


if len(sys.argv) > 1:
    im_name = sys.argv[1]
    try:
        test(im_name)
    except Exception as e:
        print(str(e))
else:
    print('Usage: \npython ImageCutter.py {filename}')
