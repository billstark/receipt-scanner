import os
import numpy as np
import cv2
from ReceiptGenerator.draw_receipt import draw_receipt_with_letter_boxes
from CNNModel.image_classifier import classify

OUT_PUT_DIR = './ReceiptProcessor/TestOutputs/'

def to_grey(img_np):
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    return img_np

def threshold(img_np):
    _, thresh = cv2.threshold(img_np, 100, 255, cv2.THRESH_BINARY)
    return thresh

def largest_box(img_np):
    processed_img = threshold(to_grey(img_np))
    _, contours, hierarchy = cv2.findContours(processed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    height, width = processed_img.shape
    lm_x = width
    tm_y = height
    rm_x = 0
    bm_y = 0
    print '==============================='
    for index, contour in enumerate(contours):
        print hierarchy[index][0]
        # print 'width: {}, height: {}'.format(width, height)
        # x, y, w, h = cv2.boundingRect(contour)
        # print 'x: {}, y: {}, w: {}, h: {}'.format(x, y, w, h)
        # if x == 0 and y == 0 and w == width and h == height:
        #     continue
        # if x < lm_x:
        #     lm_x = x
        # if x + w > rm_x:
        #     rm_x = x + w
        # if y < tm_y:
        #     tm_y = y
        # if y + h > bm_y:
        #     bm_y = y + h
    return lm_x, tm_y, rm_x - lm_x, bm_y - tm_y


img, letter_boxes = draw_receipt_with_letter_boxes()
img = np.array(img)
cv2.imwrite(OUT_PUT_DIR + 'test.png', np.array(img))

for index, letter_box in enumerate(letter_boxes):
    x, y, w, h = letter_box
    croped = img[y : y + h, x : x + w]
    bx, by, bw, bh = largest_box(croped)
    print bx, by, bw, bh
    cv2.rectangle(croped, (bx, by), (bx + bw, by + bh), (0, 0, 255), 1)
    cv2.imwrite(OUT_PUT_DIR + 'test_letters/{}.png'.format(index), croped)
