import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to input video")
args = vars(ap.parse_args())


# load our serialized model from disk
print("[INFO] loading video...")
# cap = cv2.VideoCapture('./video/bge_teste.avi')
cap = cv2.VideoCapture(args["video"])


# https://stackoverflow.com/questions/37695376/python-and-opencv-getting-the-duration-time-of-a-video-at-certain-points
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print ("[INFO] Video rate: {} fps" .format(fps))

frames = 0
ret, img_current = cap.read()
while(1):
	ret, img_current = cap.read()
	cv2.imwrite('./BGE_teste/bge_' + "%03d" % (frames,) + '.png',img_current)

	frames = frames + 1
	if frames>length-10:
		break
print ("[INFO] Number of processed frames: {}" .format(length))