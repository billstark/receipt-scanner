import cv2
import numpy

EXPECTED_WIDTH = 100
EXPECTED_HEIGHT = 32
ASPECT_RATIO = EXPECTED_WIDTH / EXPECTED_HEIGHT

def extend_along_y(img_np):
    h, w, c = img_np.shape
    extend = int((w / ASPECT_RATIO - h) / 2)
    return cv2.copyMakeBorder(img_np, extend, extend, 0, 0, cv2.BORDER_REPLICATE)

def extend_along_x(img_np):
    h, w, c = img_np.shape
    extend = int((h * ASPECT_RATIO - w) / 2)
    return cv2.copyMakeBorder(img_np, 0, 0, extend, extend, cv2.BORDER_REPLICATE)

def extend_image(img_np):
    height, width, chanel = img_np.shape
    if width / height > ASPECT_RATIO:
        return extend_along_y(img_np)
    return extend_along_x(img_np)

def resize_to_expected(img_np):
    return cv2.resize(img_np, (EXPECTED_WIDTH, EXPECTED_HEIGHT))

def standardize_image(img_np):
    height, width, chanel = img_np.shape
    return resize_to_expected(extend_image(img_np))
