import numpy as np
import cv2
from matplotlib import pyplot as plt

'''
#This method draws simple grid overthe image based on the passed step
#The pxstep controls the size of the grid
'''
def drawBasicGrid(image, pxstep, midX, midY):
    x = pxstep
    y = pxstep
    #Draw all x lines
    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=(255, 0, 255), thickness=1)
        x += pxstep

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=(255, 0, 255),thickness=1)
        y += pxstep

# Read the images to be aligned
im1 =  cv2.imread("darcy_test/darcy_394.png");
im2 =  cv2.imread("darcy_test/darcy_395.png");

im1 = im1[181+20:377-20,450+50:717-60]
im1 = cv2.resize(im1,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)

im2 = im2[181+20:377-20,450+50:717-60]
im2 = cv2.resize(im2,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)

# Convert images to grayscale
im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
 
# Find size of image1
sz = im1.shape
(H,W) = im1.shape[:2]
# Define the motion model
warp_mode = cv2.MOTION_HOMOGRAPHY
 
# Define 2x3 or 3x3 matrices and initialize the matrix to identity
if warp_mode == cv2.MOTION_HOMOGRAPHY :
    warp_matrix = np.eye(3, 3, dtype=np.float32)
else :
    warp_matrix = np.eye(2, 3, dtype=np.float32)
 
# Specify the number of iterations.
number_of_iterations = 5000;
 
# Specify the threshold of the increment
# in the correlation coefficient between two iterations
termination_eps = 1e-10;
 
# Define termination criteria
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
 
# Run the ECC algorithm. The results are stored in warp_matrix.
(cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
 
if warp_mode == cv2.MOTION_HOMOGRAPHY :
    # Use warpPerspective for Homography 
    im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
else :
    # Use warpAffine for Translation, Euclidean and Affine
    im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

'''
#Get the center point of the image
'''
midY = H//2
midX = W//2
print("Middle Y pixel", midY)
#Draw center line
cv2.line(im1, (0, midY), (W, midY), color=(0, 0, 255), thickness=1)
cv2.line(im1, (midX, 0), (midX, H), color=(0, 0, 255), thickness=1)

step = 10
y,x = np.mgrid[step/2:H:step,step/2:W:step]

# Show final results
cv2.imshow("Image 1", im1)
cv2.imshow("Image 2", im2)
cv2.imshow("Aligned Image 2", im2_aligned)
cv2.waitKey(0)