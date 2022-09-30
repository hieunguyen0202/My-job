import cv2
import matplotlib.pyplot as plt
def rescale(frame, scale=0.8):
    width = int(frame.shape[1] * scale)
    heigth = int(frame.shape[0] * scale)
    return cv2.resize(frame,(width,heigth),interpolation=cv2.INTER_AREA)

img = cv2.imread('morning.jpg')
img_scale = rescale(img)
# img_gray = cv2.cvtColor(img_scale,cv2.COLOR_RGB2GRAY)
# img_equalize = cv2.equalizeHist(img_gray)
# cv2.imshow("img_GRAY", img_gray)
# cv2.imshow("img", img_equalize)
# histogram = cv2.calcHist([img_equalize],[0],None,[256],[0,256])

img_yuv = cv2.cvtColor(img_scale,cv2.COLOR_RGB2YUV)


img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
img_equalize = cv2.cvtColor(img_yuv,cv2.COLOR_YUV2BGR)
cv2.imshow("img", img_equalize)
histogram = cv2.calcHist([img_equalize],[0],None,[256],[0,256])

plt.figure()
plt.title('Histogram')
plt.plot(histogram)
plt.show()
cv2.waitKey(0)