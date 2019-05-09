import numpy as np
import cv2 as cv
import argparse
import time
import matplotlib.pyplot as plt
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
# Load an color image in grayscale
# img = cv.imread('img/peppers.png',0)
start = time.time()
img = cv.imread(args["image"])
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image',img)slow.flv
# cv2.waitKey(0)
# cv2.destroyAllWindows()
end = time.time()
# show timing information on YOLO
print("[INFO] OpenCV processing took {:.6f} ms".format((end - start)*1000))

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()