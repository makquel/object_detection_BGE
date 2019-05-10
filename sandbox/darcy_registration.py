# from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Code for Feature Matching with FLANN tutorial.')
parser.add_argument('--input1', help='Path to input image 1.', default='box.png')
parser.add_argument('--input2', help='Path to input image 2.', default='box_in_scene.png')

args = parser.parse_args()
img_object = cv.imread('./psarp_female_ORB_pre.png', cv.IMREAD_GRAYSCALE)
img_scene = cv.imread('./psarp_female_ORB_pos.png', cv.IMREAD_GRAYSCALE)
# img_object = img_object[100:190,350:420]
# plt.imshow(img_object),plt.show()

if img_object is None or img_scene is None:
    print('Could not open or find the images!')
    exit(0)
#-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
# minHessian = 400
# detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
# keypoints_obj, descriptors_obj = detector.detectAndCompute(img_object, None)
# keypoints_scene, descriptors_scene = detector.detectAndCompute(img_scene, None)

# Initiate ORB detector
detector = cv.ORB_create()
# find the keypoints with ORB
keypoints_obj = detector.detect(img_object,None)
keypoints_scene = detector.detect(img_scene,None)
# compute the descriptors with ORB
keypoints_obj, descriptors_obj = detector.compute(img_object, keypoints_obj)
keypoints_scene, descriptors_scene = detector.compute(img_scene, keypoints_scene)
# FLANN parameters
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)
# create BFMatcher/FLANNMatcher object
# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
flann = cv.FlannBasedMatcher(index_params,search_params)
# Match descriptors.
# matches = bf.match(des1,des2)
knn_matches = flann.knnMatch(descriptors_obj,descriptors_scene,k=2)

#-- Filter out matches using the Lowe's ratio test
ratio_thresh = 0.70
good_matches = []
for m,n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)
#-- Draw matches
img_matches = np.empty((max(img_object.shape[0], img_scene.shape[0]), img_object.shape[1]+img_scene.shape[1], 3), dtype=np.uint8)



cv.drawMatches(img_object, keypoints_obj, img_scene, keypoints_scene, good_matches, img_matches, flags=2)
#-- Localize the object
obj = np.empty((len(good_matches),2), dtype=np.float32)
scene = np.empty((len(good_matches),2), dtype=np.float32)
for i in range(len(good_matches)):
    #-- Get the keypoints from the good matches
    obj[i,0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
    obj[i,1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
    scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
    scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[1]
H, warp_matrix =  cv.findHomography(obj, scene, cv.RANSAC)
#-- Get the corners from the image_1 ( the object to be "detected" )
obj_corners = np.empty((4,1,2), dtype=np.float32)
obj_corners[0,0,0] = 0
obj_corners[0,0,1] = 0
obj_corners[1,0,0] = img_object.shape[1]
obj_corners[1,0,1] = 0
obj_corners[2,0,0] = img_object.shape[1]
obj_corners[2,0,1] = img_object.shape[0]
obj_corners[3,0,0] = 0
obj_corners[3,0,1] = img_object.shape[0]
scene_corners = cv.perspectiveTransform(obj_corners, H)
colors = dict({
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0)})
thickness = 2

#-- Draw lines between the corners (the mapped object in the scene - image_2 )
cv.line(img_matches, (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])),\
    (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])), colors['red'], thickness)
cv.line(img_matches, (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])),\
    (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])), colors['red'], thickness)
cv.line(img_matches, (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])),\
    (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])), colors['red'], thickness)
cv.line(img_matches, (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])),\
    (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])), colors['red'], thickness)
#-- Show detected matches
# cv.imshow('Good Matches & Object detection', img_matches)
# cv.waitKey()
plt.imshow(img_matches),plt.show()