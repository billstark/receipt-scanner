import numpy as np
import cv2

im = cv2.imread('./trainingImages/test3.png')
outputImage = np.array(im, copy=True)
imgary = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgary, 0, 255, cv2.THRESH_OTSU)
# ret, thresh = cv2.threshold(imgary, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite('./trainingImages/thresh.png', thresh)

image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

imageTest = cv2.drawContours(outputImage, contours, -1, (0,255,0), 1)
cv2.imwrite('./trainingImages/contour.png', imageTest)

for index, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    cv2.imwrite('./trainingImages/cropedResults/testC{}.png'.format(index), outputImage[y: y + h, x: x + w])
    outputImage = cv2.rectangle(outputImage, (x, y), (x + w, y + h), (0, 255, 0), 1)

cv2.imwrite('./trainingImages/test2.png', outputImage)
