import numpy as np
import cv2
import os

IMG_PATH = './trainingImages'
CROP_RESULT_PATH = 'cropedResults'

def training_img_path(img_name):
    return os.path.join(IMG_PATH, img_name)

def cropped_result_path(img_name):
    print (os.path.join(IMG_PATH, CROP_RESULT_PATH, img_name))
    return os.path.join(IMG_PATH, CROP_RESULT_PATH, img_name)

def test():
    # Read img
    im = cv2.imread(training_img_path('line_1.png'))
    outputImage = np.array(im, copy=True)

    # Convert Color to white and black
    imgary = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Add contrast
    ret, thresh = cv2.threshold(imgary, 0, 255, cv2.THRESH_OTSU)
    # ret, thresh = cv2.threshold(imgary, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(training_img_path('thresh.png'), thresh)

    # Find contour
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contoured img
    imageContoured = cv2.drawContours(outputImage, contours, -1, (0,255,0), 1)
    cv2.imwrite(training_img_path('contour.png'), imageContoured)

    # Find and draw bounding rects for all contours
    for index, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.imwrite(cropped_result_path('testC{}.png'.format(index)), outputImage[y: y + h, x: x + w])
        outputImage = cv2.rectangle(outputImage, (x, y), (x + w, y + h), (0, 255, 0), 1)


    cv2.imwrite(training_img_path('test2.png'), outputImage)


test()