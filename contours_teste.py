import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# im = cv.imread('./bge_teste/bge_test_1.png')
# im = cv.imread('./darcy_surgery_basis.png')
# im = cv.imread('./darcy_surgery_basis.png')
im = cv.imread('./teste_1.png')

u_o = 100
v_o = 350
u_f = 450
v_f = 600
im = im[u_o:u_f,v_o:v_f]


imgray_raw = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
imgray = cv.equalizeHist(imgray_raw)
# cv.imshow("Gray scale image", imgray)
cv.imshow("images", np.hstack([imgray_raw, imgray]))
cv.waitKey(0)
# plt.imshow(imgray),plt.show()

# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

# instensities to be considered marker
# lower = np.array([0, 48, 80], dtype = "uint8")
# upper = np.array([20, 255, 255], dtype = "uint8")


# and determine the HSV pixel intensities that fall into
# the speicifed upper and lower boundaries
converted = cv.cvtColor(im, cv.COLOR_BGR2HSV)
skinMask = cv.inRange(converted, lower, upper)
# apply a series of erosions and dilations to the mask
# using an elliptical kernel
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
skinMask = cv.erode(skinMask, kernel, iterations = 2)
skinMask = cv.dilate(skinMask, kernel, iterations = 2)
 
# blur the mask to help remove noise, then apply the
# mask to the frame
skinMask = cv.GaussianBlur(skinMask, (3, 3), 0)
skin = cv.bitwise_and(im, im, mask = skinMask)
 
# show the skin in the image along with the mask
cv.imshow("images", np.hstack([im, skin]))
# cv.imshow("HSV image", skinMask)
cv.waitKey(0)

ret, thresh = cv.threshold(imgray, 48, 255, 0)
cv.imshow("Threshold image", skinMask)
cv.waitKey(0)
im2, contours, hierarchy = cv.findContours(skinMask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(im, contours, -1, (0,255,0), 3)
cv.imshow("Register image", im)
cv.waitKey(0)