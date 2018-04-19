import cv2
import numpy

EXPECTED_WIDTH = 100
EXPECTED_HEIGHT = 32
ASPECT_RATIO = EXPECTED_WIDTH / EXPECTED_HEIGHT

num = 0

def extend_along_y(img_np):
    h, w, c = img_np.shape
    extend = int((w / ASPECT_RATIO - h) / 2)
    return cv2.copyMakeBorder(img_np, extend, extend, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])

def extend_along_x(img_np):
    h, w, c = img_np.shape
    extend = int((h * ASPECT_RATIO - w) / 2)
    return cv2.copyMakeBorder(img_np, 0, 0, extend, extend, cv2.BORDER_CONSTANT, value=[255, 255, 255])

def extend_image(img_np):
    height, width, chanel = img_np.shape
    if width / height > ASPECT_RATIO:
        return extend_along_y(img_np)
    return extend_along_x(img_np)

def resize_to_expected(img_np):
    return cv2.resize(img_np, (EXPECTED_WIDTH - 4, EXPECTED_HEIGHT - 4))

def add_padding(img_np):
    return cv2.copyMakeBorder(img_np, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255, 255])

def standardize_image(img_np):
    global num
    height, width, chanel = img_np.shape
    resized = add_padding(resize_to_expected(extend_image(img_np)))
    cv2.imwrite('CRNNModel/local_utils/test/{}.png'.format(num), resized)
    num += 1
    return add_padding(resize_to_expected(extend_image(img_np)))
