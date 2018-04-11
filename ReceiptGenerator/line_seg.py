import cv2
import numpy as np
import os


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
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # Cut out lines
    lines = []
    boxes = []
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        cv2.rectangle(labelled,(x,y),( x + w, y + h ),(90,0,255),2)
        line = image[y:y+h, x:x+w]
        lines.append(line)
        boxes.append((x, y, w, h))

    return labelled, lines, boxes
