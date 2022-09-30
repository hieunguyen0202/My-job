import cv2
import matplotlib.pyplot as plt
import numpy as np
def rescale(frame, scale=0.8):
    width = int(frame.shape[1] * scale)
    heigth = int(frame.shape[0] * scale)
    return cv2.resize(frame,(width,heigth),interpolation=cv2.INTER_AREA)


img = cv2.imread("c0164df476406b9f3e7cf316f5609b49.jpg")
img_scale = rescale(img)
img_gray = cv2.cvtColor(img_scale,cv2.COLOR_RGB2GRAY)
# remove noise
img_blur = cv2.GaussianBlur(img_gray,(3,3),0)
img_edges = cv2.Canny(img_blur,100,200)

cv2.imshow("img", img)
cv2.imshow("img_edge", img_edges)
cv2.waitKey(0)