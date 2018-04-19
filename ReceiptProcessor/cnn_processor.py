import numpy as np
import cv2
import tensorflow as tf
from ReceiptGenerator.draw_receipt import scan_receipt
from CNNModel.image_classifier import classify

OUT_PUT_DIR = './ReceiptProcessor/TestOutputs/'
STANDARD_IMAGE_SIZE = 64
OCCUPATION_RATIO = 0.5

FONT = cv2.FONT_HERSHEY_SIMPLEX
SCALE = 1
COLOR = (0, 0, 255)
LINE_TYPE = 2

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
        offset = (width - height) / 2
        return add_border(img_np, offset, 0)

def extend_to_center(img_np):
    height, width, _ = img_np.shape
    length = int(height / OCCUPATION_RATIO)
    offset = (length - height) / 2
    return add_border(img_np, offset, offset)

def resize_to_standard(img_np):
    return cv2.resize(img_np, (STANDARD_IMAGE_SIZE, STANDARD_IMAGE_SIZE))

def to_grey(img_np):
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    return img_np

def threshold(img_np):
    _, thresh = cv2.threshold(img_np, 0 , 255 ,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def largest_box(img_np):
    processed_img = threshold(to_grey(img_np))
    height, width = processed_img.shape

    l_x = width
    t_y = height
    r_x = 0
    b_y = 0

    for y in range(0, height):
        for x in range(0, width):
            if processed_img[y][x] == 0:
                l_x = min(l_x, x)
                t_y = min(t_y, y)
                r_x = max(r_x, x)
                b_y = max(b_y, y)
    return l_x, t_y, (r_x - l_x), (b_y - t_y)

def process_receipt(receipt_img_np, letter_boxes):
    result = []
    for index, letter_box in enumerate(letter_boxes):
        x, y, w, h = letter_box

        try:
            croped = receipt_img_np[y : y + h, x : x + w]
            croped = add_border(croped, 5, 5)
            bx, by, bw, bh = largest_box(croped.copy())
            croped = croped[by - 4 : by + bh + 4, bx - 4 : bx + bw + 4]
            croped = extend_to_square(croped)
            croped = extend_to_center(croped)
            croped = resize_to_standard(croped)
            croped = to_grey(croped)
            _, thresh = cv2.threshold(croped, 150 , 255, cv2.THRESH_BINARY)

            thresh = np.asarray(thresh, np.float32)
            tf_image = tf.convert_to_tensor(thresh, np.float32)
            tf_image = tf.reshape(tf_image, [64*64])

            label = classify(tf_image)
            result.append((label, letter_box))

            cv2.putText(receipt_img_np, label, (x, y), FONT, SCALE, COLOR, LINE_TYPE)
            # cv2.rectangle(receipt_img_np, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.imwrite(OUT_PUT_DIR + 'test_letters/{}.png'.format(index), thresh)
        except AttributeError:
            continue

    cv2.imwrite(OUT_PUT_DIR + 'test.png', receipt_img_np)
    return result

img, letter_boxes = scan_receipt('./ReceiptProcessor/TestOutputs/IMG_1695.JPG')
new_img = np.array(img)

process_receipt(new_img, letter_boxes)
