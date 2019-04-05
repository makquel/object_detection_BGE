######## Video Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/16/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a video.
# It draws boxes and scores around the objects of interest in each frame
# of the video.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
import math
from scipy import linalg
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

import csv

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
# VIDEO_NAME = 'test.mov'
VIDEO_NAME = 'bge_teste.avi'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','label_map.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 6

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Open video file
video = cv2.VideoCapture(PATH_TO_VIDEO)
# https://stackoverflow.com/questions/37695376/python-and-opencv-getting-the-duration-time-of-a-video-at-certain-points
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = video.get(cv2.CAP_PROP_FPS)
# KF sampling rate
T = 1./fps
print ("[INFO] Video rate: {:2f} fps".format(fps))
print ("[INFO] Video length: {} frames".format(length))

# get vcap property 
# width = video.get(cv2.CV_CAP_PROP_FRAME_WIDTH)   # float
# height = video.get(cv2.CV_CAP_PROP_FRAME_HEIGHT) # float
# https://www.rapidtables.com/web/color/RGB_Color.html
colors = dict({
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "black":(0, 0, 0),
        "white":(255, 255, 255),
        "yellow":(255, 255, 0),
        "cyan":(0, 255, 255),
        "magenta":(255, 0, 255),
        "silver":(192, 192, 192),
        "gray":(128, 128, 128),
        "maroon":(128, 0, 0),
        "olive":(128, 128, 0),
        "purple":(255, 0, 255),
        "teal":(0, 128, 128),
        "navy":(0, 0, 128)})
color_key = []
for key in colors:
    color_key.append(key)
"""
State update matrices
"""
F = np.matrix( ((1, 0, T, 0),(0, 1, 0, T),(0, 0, 1, 0),(0, 0, 0, 1)) )
G = np.matrix( ((T**2/2),(T**2/2), T, T)).transpose()
H = np.matrix( ((1,0,0,0),(0,1,0,0)) ) # measurement function applied to the state estimate X_hat to get the expected next/new measurement
u = .005 #define acceleration magnitude
"""
Covariance matrices
"""
Sigma_v = .1; #process noise: the variability in how fast the Hexbug is speeding up (stdv of acceleration: meters/sec^2)
tkn_x = 1;  #measurement noise in the horizontal direction (x axis).
tkn_y = 1;  #measurement noise in the horizontal direction (y axis).
Ez = np.matrix(((tkn_x,0),(0,tkn_y)))*.5**2 
Ex = np.matrix( ((T**4/4,0,T**3/2,0),(0,T**4/4,0,T**3/2),(T**3/2,0,T**2,0),(0,T**3/2,0,T**2)) )*Sigma_v**2# Ex convert the process noise (stdv) into covariance matrix
P = Ex; # estimate of initial Hexbug position variance (covariance matrix)


bboxes_i = []
bboxes_i_1 = []
# https://stackoverflow.com/questions/10617045/how-to-create-a-fix-size-list-in-python
# bboxes_swap = [xmin, ymin, xmax, ymax]
bboxes_swap = [[0.0 , 0.0, 0.0, 0.0]]*14
swap_flag = [False]*14
frame_cnt = 0
'''
Files for debuggin KF
'''
outfile = open('./ANN_0.csv','w')
writer=csv.writer(outfile)

while(video.isOpened()):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    width = int(video.get(3))
    height = int(video.get(4))
    frame_expanded = np.expand_dims(frame, axis=0)
    start = time.time()
    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    end = time.time()
    ## https://stackoverflow.com/questions/26938799/printing-variables-in-python-3-4
    print ("[INFO] Object detection took {:.2f} ms" .format((end-start)*1000))
    # cv2.putText(frame, "Ts: {}ms".format((end-start)*1000),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Draw the results of the detection (aka 'visualize the results')
    # vis_util.visualize_boxes_and_labels_on_image_array(
    #     frame,
    #     np.squeeze(boxes),
    #     np.squeeze(classes).astype(np.int32),
    #     np.squeeze(scores),
    #     category_index,
    #     use_normalized_coordinates=True,
    #     line_thickness=2,
    #     min_score_thresh=0.35)
    bboxes_i = [] 
    # print(len(boxes[0][:]))
    for i in range(len(boxes[0][:])):
        if scores[0][i] > 0.5: #uncertainty
            bboxes_i.append(boxes[0][i])
            # print("[DEBUG] Index counter:{}".format(i))  
    M = len(bboxes_i[:])        
    print("[DEBUG] M vector size: {}".format(M))
    if (M >= 15):
        print("[WARNING] M vector size is greater that the original setpoints number")    
    min_flag = None # not necessary 
    # bboxes_swap = []
    Euc_min = 0
    if (frame_cnt > 0):
        ##HELP: comparisson between bboxes(k-1) e bboxes(k) 
        for m in range(len(bboxes_i_1[:])):
            ymin_p = bboxes_i_1[m][0]*height
            xmin_p = bboxes_i_1[m][1]*width
            ymax_p = bboxes_i_1[m][2]*height
            xmax_p = bboxes_i_1[m][3]*width 
            x_avg_p = int(xmin_p + (xmax_p-xmin_p)/2)
            y_avg_p = int(ymin_p + (ymax_p-ymin_p)/2)
            # rad_avg_p = math.sqrt(((xmax_p-xmin_p)/2)**2 + ((ymax_p-ymin_p)/2)**2)
            ## ind_cnt = 0 
            # print("[DEBUG] Vector size:{}".format(len(bboxes_i[:]))) 
            for n in range(len(bboxes_i[:])):
                ymin_q = bboxes_i[n][0]*height
                xmin_q = bboxes_i[n][1]*width
                ymax_q = bboxes_i[n][2]*height
                xmax_q = bboxes_i[n][3]*width 
                x_avg_q = int(xmin_q + (xmax_q-xmin_q)/2)
                y_avg_q = int(ymin_q + (ymax_q-ymin_q)/2)
                # rad_avg_q = math.sqrt(((xmax_q-xmin_q)/2)**2 + ((ymax_q-ymin_q)/2)**2)
                d_euc = math.sqrt(((x_avg_q-x_avg_p))**2 + ((y_avg_q-y_avg_p))**2)
                # print("[DEBUG] Euc_distance: {:.2f}".format(d_euc))
                #gravar index com a menor distancia 
                if(d_euc < 7.0):
                    min_index = n
                    min_flag = True
                    Euc_min = d_euc
                # print("[DEBUG] Index counter:{}".format(n))
            # if min_flag:
            if (min_flag and (swap_flag[min_index]==False)):
                if (min_index==0):                
                    # Predict next state of the asarp_marker with the last state and predicted motion.
                    x_hat = F*x_hat + G*u;
                    # predic_state = [predic_state; x_hat(1)] ;
                    # predict next covariance
                    P = F*P*F.T + Ex;
                    # predic_var = [predic_var; P] ;
                    # predicted measurement covariance
                    # Kalman Gain
                    K = P*H.T*linalg.inv(H*P*H.T + Ez);
                    # Update the state estimate
                    z = np.matrix( (x_avg_q, y_avg_q) ).transpose()
                    writer.writerow([x_avg_q,y_avg_q])
                    x_hat = x_hat + K*(z - H*x_hat);
                    print("predicted state for bboxes[0]: {}" .format(x_hat))
                    # update covariance estimation.
                    P = (np.identity(4) - K*H)*P;
                
                #inserir a lista bboxes_swap na posição [m] (mantendo a posição original )
                # bboxes_swap.append(bboxes_i[min_index])
                swap_flag[min_index] = True
                bboxes_swap[m] = bboxes_i[min_index]
                print("[DEBUG] Euc_distance: {:.2f}".format(Euc_min)) 
                print("[DEBUG] Index: {}".format(min_index))
                print("------------------------------------------------------------")    
            min_flag = False    
            # print("[DEBUG] Euclidean distance: {:.2f}".format(d_euc))
        for swp_i in range(len(swap_flag[:])):
            swap_flag[swp_i] = False                
    else:
        #na iteração i=0 bboxes_swap pode ser uma copia do bboxes_i ou seja bboxes_i_!
        ymin_dot = bboxes_i[0][0]*height
        xmin_dot = bboxes_i[0][1]*width
        ymax_dot = bboxes_i[0][2]*height
        xmax_dot = bboxes_i[0][3]*width 
        x_dot_avg = int(xmin_dot + (xmax_dot-xmin_dot)/2)
        y_dot_avg = int(ymin_dot + (ymax_dot-ymin_dot)/2)
        x_hat = np.matrix( (x_dot_avg, y_dot_avg, 0, 0) ).transpose()
        # bboxes_swap = bboxes_i
        for idx in range(len(bboxes_i[:])):
            print("index_cnt: {}".format(idx))
            bboxes_swap[idx] = bboxes_i[idx]




    # print("Swap vector size: {} ".format(len(bboxes_swap[:])))
    frame_cnt = frame_cnt + 1    
    print("Done processing")

    # print(bboxes_swap)
    ''' Loop só para desenho das bounding boxes'''
    # for k in range(len(bboxes_swap[:])):
    #     # if(bboxes_swap[k] == None):
    #     ymin = bboxes_swap[k][0]*height
    #     xmin = bboxes_swap[k][1]*width
    #     ymax = bboxes_swap[k][2]*height
    #     xmax = bboxes_swap[k][3]*width
    #     x_avg = int(xmin + (xmax-xmin)/2)
    #     y_avg = int(ymin + (ymax-ymin)/2)
    #     rad_avg = math.sqrt(((xmax-xmin)/2)**2 + ((ymax-ymin)/2)**2)
    #     cv2.circle(frame, (x_avg, y_avg), int(rad_avg), colors['navy'], thickness=2, lineType=8, shift=0)
    cv2.circle(frame, (x_hat[0], x_hat[1]), int(30), colors['red'], thickness=2, lineType=8, shift=0)    
    ## boxes[0][i] i-th box
    ## print(boxes[0][1])
    # https://stackoverflow.com/questions/3294889/iterating-over-dictionaries-using-for-loops
    ''' Loop só para desenho das bounding boxes'''
    for j in range(len(bboxes_swap[:])):
        ymin = bboxes_swap[j][0]*height
        xmin = bboxes_swap[j][1]*width
        ymax = bboxes_swap[j][2]*height
        xmax = bboxes_swap[j][3]*width
        x_avg = int(xmin + (xmax-xmin)/2)
        y_avg = int(ymin + (ymax-ymin)/2)
        rad_avg = math.sqrt(((xmax-xmin)/2)**2 + ((ymax-ymin)/2)**2)
        # print(x_avg, y_avg,rad_avg)
        cv2.circle(frame, (x_avg, y_avg), int(rad_avg), colors[color_key[j]], thickness=2, lineType=8, shift=0)
        cv2.putText(frame,str(j),((x_avg-5),(y_avg+5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,colors[color_key[j]],2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)
    # cv2.waitKey(1000)
    ''' Update bboxes vector for i -> i+1 '''
    ''' Analogamente i-1 -> i'''
    # bboxes_i_1 = bboxes_i
    bboxes_i_1 = bboxes_swap
    # bboxes_swap.clear()
    N = len(bboxes_i_1[:])        
    print("[DEBUG] N vector size: {}".format(N))
 
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
    frame_cnt = frame_cnt + 1
       

# Clean up
bboxes_i.clear() 
bboxes_i_1.clear() 
bboxes_swap.clear() 
video.release()
frame.release()
cv2.destroyAllWindows()
