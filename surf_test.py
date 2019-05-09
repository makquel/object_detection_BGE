# from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
from matplotlib import pyplot as plt
import time

def load_data_set(path ='path_to_your_image_set'):
    images_list = []
    for i in range(1, 400): ### Set range for proper image frame 
        # 'frame' + "%02d" % (i,) + '.png'
        # images_list.append(path +'bge_' + "%02d" % (i,) + '.png') 
        # images_list.append(path +'frame' + "%04d" % (i,) + '.jpg')
        images_list.append(path +'bge_test_' + "%d" % (i,) + '.png')
    print (images_list[0])
    return images_list

parser = argparse.ArgumentParser(description='Code for Feature Matching with FLANN tutorial.')
parser.add_argument('--input1', help='Path to input image 1.', default='box.png')
parser.add_argument('--input2', help='Path to input image 2.', default='box_in_scene.png')
args = parser.parse_args()


# images_list = load_data_set('./cyberskin_model/')
# images_list = load_data_set('./rosbag_dataset/')
images_list = load_data_set('./bge_teste/')

# img_object = cv.imread(images_list[0], cv.IMREAD_GRAYSCALE)
# Load pattern image
# img_object = cv.imread('./ncbi_test/darcy_23.png', cv.IMREAD_GRAYSCALE)
# img_object = cv.imread('./cyberskin_model/bge_01.png', cv.IMREAD_GRAYSCALE)
img_object = cv.imread('./bge_teste/bge_test_1.png', cv.IMREAD_GRAYSCALE)
# plt.imshow(img_object),plt.show()
# img_object = img_object[100:190,350:420]
# img_object = img_object[100:450,500:700]
# img_object = img_object[400:800,400:1200]
u_o = 150
v_o = 250
u_f = u_o+(380-150)
v_f = v_o+(500-250)
img_object = img_object[u_o:u_f,v_o:v_f]
# plt.imshow(img_object),plt.show()
for file in range(len(images_list)):#-1
    
    # cv.imshow("Blended warp, with padding", img_x)
    # cv.waitKey(0)
    print( "image compute: {}" .format(file) )
    img_scene = cv.imread(images_list[file], cv.IMREAD_GRAYSCALE)
    # plt.imshow(img_scene),plt.show()
    
    if img_object is None or img_scene is None:
        print('Could not open or find the images!')
        exit(0)
    #-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
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
    ratio_thresh = 0.70
    good_matches = []
    matches_cnt = 0
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
            matches_cnt = matches_cnt + 1

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
    print ("Numer of good_matches: {} " .format(matches_cnt)) 
    print (keypoints_obj[good_matches[1].trainIdx].pt[0])
    print (keypoints_obj[good_matches[1].trainIdx].pt[1])
    print (keypoints_scene[good_matches[1].queryIdx].pt[0])
    print (keypoints_scene[good_matches[1].queryIdx].pt[1])

    img_swap = img_object
    img_swap2 = img_scene
    cv.circle(img_swap,(int(keypoints_obj[good_matches[0].trainIdx].pt[0]),int(keypoints_obj[good_matches[0].trainIdx].pt[1])), 10, (0,0,255), 1)
    cv.circle(img_swap2,(int(keypoints_scene[good_matches[0].queryIdx].pt[0]),int(keypoints_scene[good_matches[0].queryIdx].pt[1])), 10, (0,0,255), 1)
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
    start = time.time()     
    H, warp_matrix =  cv.findHomography(obj, scene, cv.RANSAC)
    end = time.time()
    print ("Processed time:  {} ms" .format((end - start)*1000))
    # print file

    #-- Get the corners from the image_1 ( the object to be "detected" )
    obj_corners = np.empty((8,1,2), dtype=np.float32)
    obj_corners[0,0,0] = 0
    obj_corners[0,0,1] = 0
    obj_corners[1,0,0] = img_object.shape[1]
    obj_corners[1,0,1] = 0
    obj_corners[2,0,0] = img_object.shape[1]
    obj_corners[2,0,1] = img_object.shape[0]
    obj_corners[3,0,0] = 0
    obj_corners[3,0,1] = img_object.shape[0]
    obj_corners[4,0,0] = 0
    obj_corners[4,0,1] = img_object.shape[0]/2
    obj_corners[5,0,0] = img_object.shape[1]
    obj_corners[5,0,1] = img_object.shape[0]/2
    obj_corners[6,0,0] = img_object.shape[1]/2
    obj_corners[6,0,1] = 0
    obj_corners[7,0,0] = img_object.shape[1]/2
    obj_corners[7,0,1] = img_object.shape[0]
    scene_corners = cv.perspectiveTransform(obj_corners, H) ## It fails a lot!
    
    # print scene_corners[0,0,0]
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

    cv.line(img_matches, (int(scene_corners[4,0,0] + img_object.shape[1]), int(scene_corners[4,0,1])),\
        (int(scene_corners[5,0,0] + img_object.shape[1]), int(scene_corners[5,0,1])), colors['red'], thickness)

    cv.line(img_matches, (int(scene_corners[6,0,0] + img_object.shape[1]), int(scene_corners[6,0,1])),\
        (int(scene_corners[7,0,0] + img_object.shape[1]), int(scene_corners[7,0,1])), colors['red'], thickness)

    #-- Show detected matches
    # cv.imshow('Good Matches & Object detection', img_matches)
    # cv.waitKey()
    # plt.imshow(img_matches),plt.show()
    cv.imshow("Register image", img_matches)
    cv.waitKey(0)
    # cv.waitKey(0)