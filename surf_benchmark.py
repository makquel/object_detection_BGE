# from __future__ import print_function
import cv2 
import numpy as np
import argparse
from matplotlib import pyplot as plt
import time

colors = dict({
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0)})

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required = True, help = "Path to the image")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(args["video"])
# https://stackoverflow.com/questions/37695376/python-and-opencv-getting-the-duration-time-of-a-video-at-certain-points
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print ("Video rate: {} fps" .format(fps))

# Take first frame 
ret, image = cap.read()#skip the first frame
# image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
u_o = 120
v_o = 300
u_f = 450
v_f = 700
gray = gray[u_o:u_f,v_o:v_f]

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html
# Here I set Hessian Threshold to 400
minHessian = 400
detector = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
keypoints_1, descriptors_1 = detector.detectAndCompute(gray, None)
print ("keypoints: {}".format(len(keypoints_1)))

img2 = cv2.drawKeypoints(gray,keypoints_1,None,colors['red'],4)

plt.imshow(img2),plt.show()

frames = 0
channel_value_px = []
minHessian = 400
detector = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
while(1):
	frames = frames + 1
	ret, image = cap.read()
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = image[u_o:u_f,v_o:v_f]
	start = time. time()
	keypoints_1, descriptors_1 = detector.detectAndCompute(image, None)
	end = time. time()
	print("Execution time: {}ms".format((end - start)*1000))
	print ("keypoints: {}".format(len(keypoints_1)))

	img_kp = cv2.drawKeypoints(image,keypoints_1,None,colors['blue'],4)
	cv2.imshow('Feature-based', img_kp)
	k = cv2.waitKey(30) & 0xff
	# Now update the previous frame and previous points
	# img_prior_cropped = img_current_cropped.copy()
	print ("Processed frame: {} ".format(frames))
	if k == 27:
		break
	elif frames>length-50:
		break	
# cv2.imshow("image", gray)
# cv2.waitKey(0)