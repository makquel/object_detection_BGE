import cv2 as cv
import numpy as np
import argparse
from matplotlib import pyplot as plt
import time

# https://www.owlnet.rice.edu/~elec539/Projects97/morphjrks/laplacian.html
# img_object = cv.imread('./teste_1.png')
# img_object = cv.imread('./darcy_surgery_basis.png')
# img_object = cv.imread('./darcy_surgery_occluded.png')
img_object = cv.imread('./bge_relax.png')

img_object = cv.cvtColor(img_object, cv.COLOR_BGR2GRAY)

img_target = cv.cvtColor(img_object,cv.COLOR_GRAY2BGR)
# print img_target.shape[0]


# img_object_blurred = cv.equalizeHist(img_object)

# cv.imshow("images", np.hstack([img_object, img_object_blurred]))
# cv.waitKey(0)

# u_o = 100
# v_o = 300
# u_f = 450
# v_f = 600
ROI_offset = -40

ROI_radius = 160 #pixel size of the search radius
# ROI_radius = 90 #pixel size of the search radius
cv.circle(img_target,(int(img_target.shape[1]/2),int(img_target.shape[0]/2)+ROI_offset), ROI_radius, (255,0,0), thickness=1, lineType=8, shift=0)
plt.imshow(img_target),plt.show()

v_o = int(img_target.shape[1]/2)-ROI_radius
u_o = int(img_target.shape[0]/2)-ROI_radius+ROI_offset
v_f = int(img_target.shape[1]/2)+ROI_radius
u_f = int(img_target.shape[0]/2)+ROI_radius+ROI_offset

img_object = cv.equalizeHist(img_object[u_o:u_f,v_o:v_f])

# img_scene = cv.imread('./teste_1.png', cv.IMREAD_GRAYSCALE)
# img_scene = cv.imread('./darcy_surgery_basis.png', cv.IMREAD_GRAYSCALE)
# img_scene = cv.imread('./darcy_surgery_occluded.png', cv.IMREAD_GRAYSCALE)
img_scene = cv.imread('./bge_contract.png', cv.IMREAD_GRAYSCALE)
# img_scene_blurred = cv.equalizeHist(img_scene)
img_scene = cv.equalizeHist(img_scene[u_o:u_f,v_o:v_f])

minHessian = 400
detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
keypoints_obj, descriptors_obj = detector.detectAndCompute(img_object, None)
keypoints_scene, descriptors_scene = detector.detectAndCompute(img_scene, None)
#-- Step 2: Matching descriptor vectors with a FLANN based matcher
# Since SURF is a floating-point descriptor NORM_L2 is used
matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
# https://stackoverflow.com/questions/16996800/what-does-the-distance-attribute-in-dmatches-mean/16997140
knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)
#-- Filter matches using the Lowe's ratio test
# https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work
ratio_thresh = 0.70
good_matches = []
matches_cnt = 0
for m,n in knn_matches:
	if m.distance < ratio_thresh * n.distance:
		good_matches.append(m)
		matches_cnt = matches_cnt + 1

# # Detect ORB features and compute descriptors.
# orb = cv.ORB_create()
# keypoints_obj, descriptors_obj = orb.detectAndCompute(img_object, None)
# keypoints_scene, descriptors_scene = orb.detectAndCompute(img_scene, None)
# #keep only the keypoints within the radius of ROI

# #-- Step 2: Matching descriptor vectors with a FLANN based matcher
# FLANN_INDEX_LSH = 6
# index_params= dict(algorithm = FLANN_INDEX_LSH,
#                    table_number = 6, # 12
#                    key_size = 12,     # 20
#                    multi_probe_level = 1) #2
# search_params = dict(checks=50) # or pass empty dictionary
# # create BFMatcher/FLANNMatcher object
# flann = cv.FlannBasedMatcher(index_params,search_params)
# # Match descriptors.
# # matches = bf.match(des1,des2)
# matches = flann.knnMatch(descriptors_obj,descriptors_scene,k=2)

# good_matches = matches
# # Brute Force Matching
# # bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# # good_matches = bf.match(descriptors_obj, descriptors_scene)
# # good_matches = sorted(good_matches, key = lambda x:x.distance)
# matches_cnt = 0
# for k in good_matches:
# 	matches_cnt = matches_cnt + 1

# https://stackoverflow.com/questions/37514946/opencv-c-dmatch-is-the-descriptor-index-the-same-as-its-corresponding-k        
'''
DMatch.distance - Distance between descriptors. The lower, the better it is.
DMatch.trainIdx - Index of the descriptor in train descriptors(1st image)
DMatch.queryIdx - Index of the descriptor in query descriptors(2nd image)
DMatch.imgIdx - Index of the train image.
'''
# for match in good_matches:
#     p1 = keypoints_obj[match.trainIdx].pt
#     p2 = keypoints_scene[match.queryIdx].pt
# # print p1   

# print (good_matches[1].queryIdx)
# print (good_matches[1].trainIdx)   
print "Numer of good_matches: %2d " % matches_cnt 
print (keypoints_obj[good_matches[0].trainIdx].pt[0])
print (keypoints_obj[good_matches[0].trainIdx].pt[1])
print (keypoints_scene[good_matches[0].queryIdx].pt[0])
print (keypoints_scene[good_matches[0].queryIdx].pt[1])

img_swap = img_object
img_swap2 = img_scene
# cv.circle(img_swap,(int(keypoints_obj[good_matches[0].trainIdx].pt[0]),int(keypoints_obj[good_matches[0].trainIdx].pt[1])), 10, (0,0,255), 1)
# cv.circle(img_swap2,(int(keypoints_scene[good_matches[0].queryIdx].pt[0]),int(keypoints_scene[good_matches[0].queryIdx].pt[1])), 10, (0,0,255), 1)
# cv.imshow("swap_image", img_swap)
# cv.waitKey(0)
# plt.imshow(img_swap),plt.show()

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
# start = time.time()     
# H, warp_matrix =  cv.findHomography(obj, scene, cv.RANSAC)
# end = time.time()
# print "Processed time:  %0.5f s" % ((end - start)*1000)
# cv.imshow("Matches", img_matches)
# cv.waitKey(0)
cv.destroyAllWindows()

plt.imshow(img_matches),plt.show()