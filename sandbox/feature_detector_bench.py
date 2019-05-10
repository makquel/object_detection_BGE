## https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
## https://docs.opencv.org/3.4/d7/dff/tutorial_feature_homography.html
## https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
# from __future__ import print_function
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time
import math
import argparse
# img = cv.imread('./psarp_female_ORB_test.png',0)
# img = cv.imread('./psarp_female_ORB_pre.png',0)
# img1 = cv.imread('./psarp_female_ORB_pre.png',0)# queryImage
# img2 = cv.imread('./psarp_female_ORB_pos.png',0) # trainImage
# img1 = cv.imread('./color_raw.png',0)# 
# img1 = cv.imread('./high_color_raw_pusher.png',0)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading image...")

img1 = cv.imread(args["image"]) #0 stands for gray scale image
# img1_roi = img1[400:1000,400:1200]
# u_o = 150
# v_o = 400
# u_f = u_o+250
# v_f = v_o+200
# u_o = 400+200
# v_o = 400+300
# u_f = u_o+400
# v_f = v_o+800
u_o = 100
v_o = 300
u_f = 450
v_f = 600
img1_roi = img1[u_o:u_f,v_o:v_f]
roi_grey = cv.cvtColor(img1_roi, cv.COLOR_BGR2GRAY) # convert to grayscale

f, (ax1, ax2) = plt.subplots(1, 2) # create subplots
# Initiate ORB detector
# orb = cv.ORB_create()
# find the keypoints with ORB
# kp1 = orb.detect(img1,None)
# compute the descriptors with ORB
# kp1, des1 = orb.compute(img1, kp1)
# print kp1[0].pt

minHessian = 400
detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
kp1, des1 = detector.detectAndCompute(roi_grey, None)
# draw only keypoints location,not size and orientation
img1_kp = cv.drawKeypoints(img1_roi, kp1, None, color=(0,255,0), flags=0)
ax1.imshow(img1_kp)

pts = np.asarray([[p.pt[0], p.pt[1]] for p in kp1])
cols = pts[:,0]+v_o
rows = pts[:,1]+u_o
colors = dict({
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0)})
thickness = 2
cv.rectangle(img1,(v_o,u_o),(v_f,u_f),colors['red'],thickness)
ax2.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
ax2.scatter(cols, rows)

plt.show()