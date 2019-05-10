import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('box.png',0)          # queryImage
img2 = cv2.imread('box_in_scene.png',0) # trainImage

# Initiate SIFT detector
orb = cv2.ORB()
# find the keypoints with ORB
kp1 = orb.detect(img1,None)
kp2 = orb.detect(img2,None)
# compute the descriptors with ORB
kp1, des1 = orb.compute(img1, kp1)
kp2, des2 = orb.compute(img2, kp2)
# draw only keypoints location,not size and orientation
img1_kp = cv.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
img2_kp = cv.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)

plt.imshow(img1_kp), plt.show()
plt.imshow(img2_kp), plt.show()