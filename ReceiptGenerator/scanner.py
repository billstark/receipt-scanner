from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

def scan(image):
    # Compute the ratio of the old height to the new height, clone it, and resize it
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height = 500)

    # Grayscale, blur, and find edges in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # Find the contours, keep the largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if contour has four points, can assume that it is the bounds
        if len(approx) == 4:
            screen_cnt = approx
            break

    # Four point transform
    warped = four_point_transform(orig, screen_cnt.reshape(4, 2) * ratio)

    # # Grayscale, 'black and white' paper effect
    # warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # threahold = threshold_local(warped, 11, offset = 10, method = "gaussian")
    # warped = (warped > threahold).astype("uint8") * 255

    return warped
