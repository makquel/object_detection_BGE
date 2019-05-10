import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('./img/Urban/frame07.png')
img2 = cv2.imread('./img/Urban/frame08.png')
# print img.shape
# print img.dtype
# cropped = img[181:377,450:717]
# plt.imshow(cropped),plt.show()

img1_ = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2_ = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Take first frame and find corners in it
p0 = cv2.goodFeaturesToTrack(img1_,mask = None, **feature_params)
# corners = np.int0(corners)
p0 = np.array(p0)
print (p0)
print (p0.shape)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
corners_subpix = cv2.cornerSubPix(img1_, p0, (3,3), (-1,1), criteria)

for i in corners_subpix:
    x,y = i.ravel()
    cv2.circle(img1_,(x,y),3,255,-1)

# calculate optical flow
p1, st, err = cv2.calcOpticalFlowPyrLK(img1_, img2_, p0, None, **lk_params)

# Select good points
good_new = p1[st==1]
good_old = p0[st==1]

print p1
print p1.shape
plt.imshow(img1_),plt.show()