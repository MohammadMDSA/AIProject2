import time

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math as M

plt.xticks([]), plt.yticks([])

print(cv.__version__)
img = cv.imread("files/test1.jpg")

cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()

b = img.copy()
b[:, :, 1] = 0
b[:, :, 2] = 0

cv.imshow('blue', b)
cv.waitKey(0)
cv.destroyAllWindows()

r, g, b = cv.split(img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)
cv.waitKey(0)
cv.destroyAllWindows()

blur = cv.GaussianBlur(gray, (9, 9), 0)
cv.imshow('blur', blur)
cv.waitKey(0)
cv.destroyAllWindows()

(h, w) = img.shape[:2]
rotation = cv.getRotationMatrix2D((w / 2, h / 2), 90, 1.0)
rotated = cv.warpAffine(img, rotation, (h, w))
cv.imshow('rotated', rotated)
cv.waitKey(0)
cv.destroyAllWindows()

cut_picture = img[:, : M.floor(w / 2), :]
cv.imshow('cut', cut_picture)
cv.waitKey(0)
cv.destroyAllWindows()

edges = cv.Canny(img, 100, 200)
cv.imshow('edge', edges)
cv.waitKey(0)
cv.destroyAllWindows()

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

sure_bg = cv.dilate(opening, kernel, iterations=3)

dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1
# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

new_img = img.copy()

markers = cv.watershed(new_img, markers)
new_img[markers == -1] = [255, 0, 0]

cv.imshow('segmentation', new_img)
cv.waitKey(0)
cv.destroyAllWindows()

face_cascade = cv.CascadeClassifier('files/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('files/haarcascade_eye.xml')

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()

vidcap = cv.VideoCapture("files/test.avi")

for i in range(4):
    success, image = vidcap.read()
    print(success)
    cv.imshow('capture', image)
    # time.sleep(1)
    cv.waitKey(500)
cv.destroyAllWindows()
