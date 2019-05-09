import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import time

# img = cv.imread('./teste_2.png')
img = cv.imread('./darcy_surgery_basis.png')
# img = cv.bitwise_not(img)

img_cropped = img[100:400,300:600]
img_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# cv.circle(img_HSV,(460,260), 10, (0,0,255), 1)
print (img_HSV[460,250])
# plt.imshow(img_HSV),plt.show()
cv.imshow("Result HSV", img_HSV)
cv.waitKey(0)
img_LAB = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow("Result LAB", img_HSV)
cv.waitKey(0)
bgr = [83, 77, 122]
thresh = 44
MAX_FEATURES = 500
'''
Note: For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]. Different software use different scales. 
So if you are comparing OpenCV values with them, you need to normalize these ranges.
Hue values of basic colors
Orange 0-22
Yellow 22- 38
Green 38-75
Blue 75-130
Violet 130-160
Red 160-179
'''
tol = 5
hsv = cv.cvtColor( np.uint8([[bgr]] ), cv.COLOR_BGR2HSV)[0][0]
# lower_bound = np.array([hsv[0] - thresh, hsv[1] - thresh, hsv[2] - thresh])
# upper_bound = np.array([hsv[0] + thresh, hsv[1] + thresh, hsv[2] + thresh])
lower_bound = np.array([0, 0, 200], dtype = "uint8")
upper_bound = np.array([180, 255,255], dtype = "uint8")

maskHSV = cv.inRange(img_HSV, lower_bound, upper_bound)
resultHSV = cv.bitwise_and(img_HSV, img_HSV, mask = maskHSV)

# lab = cv.cvtColor( np.uint8([[bgr]] ), cv.COLOR_BGR2LAB)[0][0]
# minLAB = np.array([lab[0] - thresh, lab[1] - thresh, lab[2] - thresh])
# maxLAB = np.array([lab[0] + thresh, lab[1] + thresh, lab[2] + thresh])
 
# maskLAB = cv.inRange(img_LAB, minLAB, maxLAB)
# resultLAB = cv.bitwise_and(img_LAB, img_LAB, mask = maskLAB)

cv.imshow("Result HSV seg", cv.cvtColor(resultHSV, cv.COLOR_HSV2BGR))
cv.waitKey(0)
im_GRAY = cv.cvtColor(cv.cvtColor(resultHSV, cv.COLOR_HSV2BGR), cv.COLOR_BGR2GRAY)
cv.imshow("Result LAB seg", im_GRAY)
cv.waitKey(0)
# Detect ORB features and compute descriptors.
orb = cv.ORB_create(MAX_FEATURES)
keypoints1, descriptors1 = orb.detectAndCompute(cv.cvtColor(cv.cvtColor(img, cv.COLOR_HSV2BGR), cv.COLOR_BGR2GRAY), None)
# print keypoints1[0]
im_orb = cv.drawKeypoints(img,keypoints1,None,color=(0,255,0), flags=0)
cv.imshow("ORB keypoints", im_orb)
cv.waitKey(0)
cv.destroyAllWindows()