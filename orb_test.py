## https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
## https://docs.opencv.org/3.4/d7/dff/tutorial_feature_homography.html
## https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
# from __future__ import print_function
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time
import math




# img = cv.imread('./psarp_female_ORB_test.png',0)
# img = cv.imread('./psarp_female_ORB_pre.png',0)
# img1 = cv.imread('./psarp_female_ORB_pre.png',0)# queryImage
# img2 = cv.imread('./psarp_female_ORB_pos.png',0) # trainImage
# img1 = cv.imread('./box.png',0)# queryImage
# img2 = cv.imread('./box_in_scene.png',0) # trainImage

ROI_offset = -40
ROI_radius = 160 #pixel size of the search radius
# ROI_radius = 90 #pixel size of the search radius
img1 = cv.imread('./bge_relax.png',0)
v_o = int(img1.shape[1]/2)-ROI_radius
u_o = int(img1.shape[0]/2)-ROI_radius+ROI_offset
v_f = int(img1.shape[1]/2)+ROI_radius
u_f = int(img1.shape[0]/2)+ROI_radius+ROI_offset
img1 = cv.equalizeHist(img1[u_o:u_f,v_o:v_f])
img2 = cv.imread('./bge_contract.png',0)
img2 = cv.equalizeHist(img2[u_o:u_f,v_o:v_f])

# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints with ORB
kp1 = orb.detect(img1,None)
kp2 = orb.detect(img2,None)
# compute the descriptors with ORB
kp1, des1 = orb.compute(img1, kp1)
kp2, des2 = orb.compute(img2, kp2)
# print kp1[0].pt

# FLANN parameters
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict(checks=50)   # or pass empty dictionary
# draw only keypoints location,not size and orientation
img1_kp = cv.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
img2_kp = cv.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)
# x = 200
# y = 50
# w = 250
# h = 250
# cv.rectangle(img2, (x, y), (x+w, y+h), (255, 0, 0), 2)
# plt.imshow(img1_kp), plt.show()
# plt.imshow(img2_kp), plt.show()

# flann = cv.FlannBasedMatcher(index_params,search_params)

# create BFMatcher/FLANNMatcher object
# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
flann = cv.FlannBasedMatcher(index_params,search_params)
# Match descriptors.
# matches = bf.match(des1,des2)
matches = flann.knnMatch(des1,des2,k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
#-- Filter matches using the Lowe's ratio test
# ratio test as per Lowe's paper
matches_cnt = 0
# for i,(m,n) in enumerate(matches):
#     if m.distance < 0.65*n.distance:
#         matchesMask[i]=[1,0]
#         matches_cnt = matches_cnt + 1

print ("Numer of good_matches: {}" .format(matches_cnt)) 
      

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

# Sort them in the order of their distance.
# matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
# img_match = cv.drawMatches(img1,kp1,img2,kp2,matches[:10], None,draw_params)
img_match = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
plt.imshow(img_match),plt.show()

#-- Localize the object
obj = np.empty((len(matches),2), dtype=np.float32)
scene = np.empty((len(matches),2), dtype=np.float32)
for i in range(len(matches)):
    #-- Get the keypoints from the good matches
    obj[i,0] = kp1[i].pt[0]
    obj[i,1] = kp1[i].pt[1]
    scene[i,0] = kp2[i].pt[0]
    scene[i,1] = kp2[i].pt[1]
# print obj     

# Use homography
# Define the motion model
warp_mode = cv.MOTION_HOMOGRAPHY
# warp_mode = cv.MOTION_AFFINE
 
# Define 2x3 or 3x3 matrices and initialize the matrix to identity
if warp_mode == cv.MOTION_HOMOGRAPHY :
    warp_matrix = np.eye(3, 3, dtype=np.float32)
else :
    warp_matrix = np.eye(2, 3, dtype=np.float32)
# warp_matrix.shape
# Specify the number of iterations.
number_of_iterations = 5000;
print ("Running with {} iterations" .format(number_of_iterations))
# Specify the threshold of the increment
# in the correlation coefficient between two iterations
termination_eps = 1e-10;
# Define termination criteria
criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
"""
	Run the ECC algorithm. The results are stored in warp_matrix.
	"""
start = time.time() 
# (cc, warp_matrix) = cv.findTransformECC (obj, scene, warp_matrix, warp_mode, criteria)
(H, warp_matrix) =  cv.findHomography(obj, scene, cv.RANSAC,  5.0)
end = time.time()
print ("Processed time: {} s" .format((end - start)*1000))

# see https://ch.mathworks.com/help/images/examples/find-image-rotation-and-scale-using-automated-feature-matching.html for details
ss = H[0, 1]
sc = H[0, 0]
scaleRecovered = math.sqrt(ss * ss + sc * sc)
thetaRecovered = math.atan2(ss, sc) * 180 / math.pi
print ("MAP: Calculated scale difference: %.2f, " "Calculated rotation difference: {}" .format(scaleRecovered, thetaRecovered))

#-- Get the corners from the image_1 ( the object to be "detected" )
obj_corners = np.empty((4,1,2), dtype=np.float32)
obj_corners[0,0,0] = 0
obj_corners[0,0,1] = 0
obj_corners[1,0,0] = img1.shape[1]
obj_corners[1,0,1] = 0
obj_corners[2,0,0] = img1.shape[1]
obj_corners[2,0,1] = img1.shape[0]
obj_corners[3,0,0] = 0
obj_corners[3,0,1] = img1.shape[0]

scene_corners = cv.perspectiveTransform(obj_corners, H)

#-- Draw lines between the corners (the mapped object in the scene - image_2 )
cv.line(img2, (int(scene_corners[0,0,0] + img1.shape[1]), int(scene_corners[0,0,1])),\
    (int(scene_corners[1,0,0] + img1.shape[1]), int(scene_corners[1,0,1])), (0,0,255), 4)
cv.line(img2, (int(scene_corners[1,0,0] + img1.shape[1]), int(scene_corners[1,0,1])),\
    (int(scene_corners[2,0,0] + img1.shape[1]), int(scene_corners[2,0,1])), (0,0,255), 4)
cv.line(img2, (int(scene_corners[2,0,0] + img1.shape[1]), int(scene_corners[2,0,1])),\
    (int(scene_corners[3,0,0] + img1.shape[1]), int(scene_corners[3,0,1])), (0,0,255), 4)
cv.line(img2, (int(scene_corners[3,0,0] + img1.shape[1]), int(scene_corners[3,0,1])),\
    (int(scene_corners[0,0,0] + img1.shape[1]), int(scene_corners[0,0,1])), (0,0,255), 4)
#-- Show detected matches
cv.imshow('Good Matches & Object detection', img2)
cv.waitKey()

# height, width = img2.shape
# if warp_mode == cv.MOTION_HOMOGRAPHY :
# 	# Use warpPerspective for Homography 
#     im2_aligned = cv.warpPerspective (img2, H, (width, height), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
# else :
#     # Use warpAffine for Translation, Euclidean and Affine
#     im2_aligned = cv.warpAffine(img2, warp_matrix, (width, height), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP);


# plt.imshow(im2_aligned),plt.show()