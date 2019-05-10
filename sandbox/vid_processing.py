import numpy as np
import cv2
from matplotlib import pyplot as plt

def draw_flow(img,flow,step=16,vector_scale=5.):
    # print step 
    h,w = img.shape[:2] 
    y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1) 
    fx,fy = flow[y,x].T 

    # create line endpoints 
    # vector_scale = 5.0
    lines = np.vstack([x,y,x+fx*vector_scale,y+fy*vector_scale]).T.reshape(-1,2,2) 
    lines = np.int32(lines) 
    # print lines.shape
    U, V = flow.shape[:2]
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    ang = (ang + np.pi) * (180. / np.pi)
    HSV = np.zeros((U, V, 3), np.uint8)
    # hue corresponds to direction
    HSV[...,0] = ang/2
    HSV[...,1] = 2**8-1
    # value corresponds to magnitude
    HSV[...,2] = np.minimum(mag*4*10, 255)
    BGR = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)
    # print mag.shape
    # print BGR[5,405,0]
    # print BGR[5,405,1]
    # print BGR[5,405,2]
    # create image and draw 
    vis = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR) 
    draw_thresh = 2
    proc_pixels = 0
    for (x1,y1),(x2,y2) in lines: 
    	# print x1, y1
    	R = BGR[y1,x1,2]
    	G = BGR[y1,x1,1]
    	B = BGR[y1,x1,0]
    	color = np.array((B,G,R))
    	# http://answers.opencv.org/question/185393/typeerror-scalar-value-for-argument-color-is-not-numeric/
    	color = np.array((np.asscalar(np.int16(color[0])),np.asscalar(np.int16(color[1])),np.asscalar(np.int16(color[2]))))
    	
    	if ((abs(x2 - x1) > draw_thresh or abs(y2 - y1) > draw_thresh) and (abs(x2 - x1) < draw_thresh*2. or abs(y2 - y1) < draw_thresh*2.)):
    		cv2.line(vis,(x1,y1),(x2,y2),color,1)
    		cv2.circle(vis,(x1,y1),1,color, -1)
    		proc_pixels = proc_pixels + 1

	# print "processed_pixels: %0.5d" % proc_pixels         	 
    return vis

def draw_hsv_flow(flow, scale=10, sat=(2**8)-1):
	U, V = flow.shape[:2]
	# convert from cartesian to polar
	mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
	ang = (ang + np.pi) * (180. / np.pi);
	# create HSV image for flow visualization
	# (8-bit: 0 <= H <= 180, 0 <= S,V <= 255)
	HSV = np.zeros((U, V, 3), np.uint8)
	# hue corresponds to direction
	HSV[...,0] = ang/2
	HSV[...,1] = 2**8-1
	# value corresponds to magnitude
	scale = 10
	HSV[...,2] = np.minimum(mag*4*scale, 255)
	BGR = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)
	return BGR

cap = cv2.VideoCapture('./video/trecho_estimulador.mp4')
# https://stackoverflow.com/questions/37695376/python-and-opencv-getting-the-duration-time-of-a-video-at-certain-points
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print ("Video rate: {} fps" .format(fps))

# Take first frame 
ret, img_prior = cap.read()#skip the first frame
ret, img_prior = cap.read()
img_prior = cv2.cvtColor(img_prior, cv2.COLOR_BGR2GRAY)
img_prior = cv2.GaussianBlur(img_prior,(3,3),1.5)
img_prior_cropped = img_prior[175:431,550:814]
img_prior_cropped = cv2.resize(img_prior_cropped,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
# img_prior_cropped = cv2.equalizeHist(img_prior_cropped)
# res = cv2.resize(img_prior_cropped,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
# cv2.imshow('resized', res)
frames = 0
channel_value_px = []
while(1):
	frames = frames + 1
	ret, img_current = cap.read()
	img_current = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)
	img_current = cv2.GaussianBlur(img_current,(3,3),1.5)
	img_current_cropped = img_current[175:431,550:814]
	img_current_cropped = cv2.resize(img_current_cropped	,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
	# img_current_cropped = cv2.equalizeHist(img_current_cropped)
	flow = cv2.calcOpticalFlowFarneback(img_prior_cropped, img_current_cropped, flow=None,pyr_scale=0.5,levels=1,winsize=10,iterations=2,poly_n=7,poly_sigma=1.5,flags=0)
	channel_value_px.append(img_current_cropped[200,200])
	# print flow[5][5][1]
	# U, V = flow.shape[:2]
	# for i in range(0,U):
	# 	for j in range(0,V):
	# 		flow[i][j][0] = 10

	# cv2.imshow('flow_field',draw_hsv_flow(flow, scale=10, sat=(2**8)-1))
	# cv2.imshow('I_t',img_current)
	
	cv2.imshow('flow_field', draw_flow(img_prior_cropped, flow,step=8, vector_scale=5.))
	k = cv2.waitKey(30) & 0xff
	# Now update the previous frame and previous points
	img_prior_cropped = img_current_cropped.copy()
	print ("Processed frame: {} ".format(frames))
	if k == 27:
		break
	elif frames>length-10:
		break	

# print channel_value_px
frame_index = np.linspace(0, frames-1, frames)
# print frame_index
# print frame_index.shape
# print np.asarray(channel_value_px).shape
plt.plot(frame_index, np.asarray(channel_value_px), label='[200,200]')

# print frame_index
plt.show()
# print "Size of video: %0.5d frames" % frames
cap.release()
cv2.destroyAllWindows()