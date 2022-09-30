

import numpy as np
import cv2

img = cv2.imread('image\\bcf-9.jpg')
img_org = img
cv2.imshow("img_origin", img_org)
img_gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
lines = cv2.HoughLines(edges, 1, np.pi /180, 200)

for line in lines:
    ro, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * ro
    y0 = b * ro
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * a)
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow("img", img)

cv2.waitKey(0)