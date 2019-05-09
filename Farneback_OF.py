import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

def draw_flow(im,flow,step=16):
    print (step) 
    h,w = im.shape[:2] 
    y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1) 
    fx,fy = flow[y,x].T 
    # create line endpoints 
    vector_scale = 10.0
    lines = np.vstack([x,y,x+fx*vector_scale,y+fy*vector_scale]).T.reshape(-1,2,2) 
    lines = np.int32(lines) 
    # create image and draw 
    vis = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR) 
    for (x1,y1),(x2,y2) in lines: 
        cv2.line(vis,(x1,y1),(x2,y2),(0,0,255),1) 
        cv2.circle(vis,(x1,y1),1,(0,0,255), -1) 
    return vis


def nothing(x):
    pass    

"""
Load images from dataset
"""
# img1 = cv2.imread('./img/img1.png')
# img2 = cv2.imread('./img/img2.png')
# img1 = cv2.imread('./img/synth_img_1.png')
# img2 = cv2.imread('./img/synth_img_1.png')
img1 = cv2.imread('./img/darcy_2.png')
img2 = cv2.imread('./img/darcy_3.png')
X, Y, channel = img1.shape
# print X
hsv = np.zeros((X, Y, 3))
# cap = cv2.VideoCapture("video/trecho_estimulador.flv")

print (img1.shape)
print (img1.dtype)
"""
ROI user defined
"""
img1_cropped = img1[175:431,550:814]
img2_cropped = img2[175:431,550:814]
cv2.imshow('img1_cropped', img1_cropped)
"""
Color conversion
"""
# img1_ = cv2.cvtColor(img1_cropped,cv2.COLOR_BGR2GRAY)
# img2_ = cv2.cvtColor(img1_cropped,cv2.COLOR_BGR2GRAY)

img1_ = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2_ = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

"""
Gaussian smooth
"""
img1_ = cv2.GaussianBlur(img1_,(3,3),1.5)
img2_ = cv2.GaussianBlur(img2_,(3,3),1.5)

"""
Farneback
"""
# grabs the initial time
t0 = time.time()
flow = cv2.calcOpticalFlowFarneback(img1_, img2_, flow=None,pyr_scale=0.5,levels=1,winsize=10,iterations=2,poly_n=7,poly_sigma=1.5,flags=0)
# flow = cv2.medianBlur(flow, 5)
# takes the final time
t1 = time.time()
print ("OF calculation took {} ms".format(((t1-t0)*1000)))

# Slice interval for quiver
slice_interval = 4
# Slicer index for smoother quiver plot
# General note: Adjust the slice interval and scale accordingly to get the required arrow size.
# Also, the units, and angles units are also responsible.
skip = (slice(None, None, slice_interval), slice(None, None, slice_interval))

x1,x2 = img1_.shape[:2]
U, V = flow.shape[:2]
X1, X2 = np.meshgrid(x1,x2)
# convert from cartesian to polar
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
ang = (ang + np.pi) * (180. / np.pi);
# create HSV image for flow visualization
# (8-bit: 0 <= H <= 180, 0 <= S,V <= 255)
HSV = np.zeros((U, V, 3), np.uint8)
# hue corresponds to direction
HSV[...,0] = ang/2
HSV[...,1] = 2**7
# value corresponds to magnitude
scale = 10
HSV[...,2] = np.minimum(mag*4*scale, 255)

# BGR = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)

plt.hist(HSV[...,0].ravel(),360,[0,360]); plt.show()
# print np.ravel(HSV[...,0])

cv2.namedWindow('flow_field')
# create trackbars for color change
cv2.createTrackbar('value','flow_field',0,255,nothing)
while(1):
    BGR = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)
    cv2.imshow('flow_field', BGR)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    # get current positions of four trackbars
    value = cv2.getTrackbarPos('value','flow_field')
    HSV[...,1] = value


cv2.imshow('flow', draw_flow(img1_, flow,step=10))
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.figure()
# Quiver = plt.quiver(X1[skip],X2[skip],U[skip],V[skip],units='height')
Quiver = plt.quiver(X1,X2,U,V)
plt.title("OF velocity vectors")
plt.xlabel("u")
plt.ylabel("v")
plt.quiverkey(Quiver,X=1.01, Y=1.01, U=10,label='Quiver key, length = 10', labelpos='E')
plt.draw()
plt.grid()
# plt.show()

