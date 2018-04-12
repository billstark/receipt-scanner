import os
import numpy as np
import cv2
from ReceiptGenerator.draw_receipt import draw_receipt_with_letter_boxes
from CNNModel.image_classifier import classify

OUT_PUT_DIR = './ReceiptProcessor/TestOutputs/'

def add_border(img_np, top_and_bottom, left_and_right):
    return cv2.copyMakeBorder(img_np, top = top_and_bottom, \
        bottom = top_and_bottom, left = left_and_right, \
        right = left_and_right, borderType = cv2.BORDER_CONSTANT, \
        value = [255, 255, 255])

def extend_to_square(img_np):
    height, width, _ = img_np.shape
    if height > width:
        offset = (height - width) / 2
        return add_border(img_np, 0, offset)
    else:
        return add_border(img_np, offset, 0)

def to_grey(img_np):
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    return img_np

def threshold(img_np):
    _, thresh = cv2.threshold(img_np, 100, 255, cv2.THRESH_BINARY)
    return thresh

def largest_box(img_np):
    processed_img = threshold(to_grey(img_np))
    height, width = img_np.shape

    l_x = width
    t_y = height
    r_x = 0
    b_y = 0

    for y in range(0, height):
        for x in range(0, width):
            if img_np[y][x] == 0:
                l_x = min(l_x, x)
                t_y = min(t_y, y)
                r_x = max(r_x, x)
                b_y = max(b_y, y)
    return l_x, t_y, (r_x - l_x), (b_y - t_y)


img, letter_boxes = draw_receipt_with_letter_boxes()
img = np.array(img)
cv2.imwrite(OUT_PUT_DIR + 'test.png', np.array(img))

for index, letter_box in enumerate(letter_boxes):
    x, y, w, h = letter_box
    croped = img[y : y + h, x : x + w]
    croped = add_border(croped, 1, 1)
    bx, by, bw, bh = largest_box(croped)
    croped = croped[by : by + bh, bx : bx + bw]
    height, width, channel = croped.shape
    croped = extend_to_square(croped)

    cv2.rectangle(croped, (bx, by), (bx + bw, by + bh), (0, 0, 255), 1)
    cv2.imwrite(OUT_PUT_DIR + 'test_letters/{}.png'.format(index), croped)
