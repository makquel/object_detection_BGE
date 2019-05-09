import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import time

# img = cv.imread('./img/pixil-frame-0.png')
# plt.imshow(img)
# plt.show()
# res = cv.resize(img,None,fx=60, fy=60, interpolation = cv.INTER_CUBIC)
# res = cv.medianBlur(res, 5)
# cv.imshow('flow', res)
# cv.waitKey(0)
# cv.destroyAllWindows()

# delta = 0.025
# x = y = np.arange(-3.0, 3.0, delta)
# X, Y = np.meshgrid(x, y)
# Z1 = np.exp(-X**2 - Y**2)
# Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
# Z = (Z1 - Z2) * 2

# fig, ax = plt.subplots()
# im = ax.imshow(Z, interpolation='bilinear', cmap=cm.RdYlGn,
#                origin='lower', extent=[-3, 3, -3, 3],
#                vmax=abs(Z).max(), vmin=-abs(Z).max())

# plt.show()
def draw_flow(im,flow,step=16):
    print (step) 
    h,w = im.shape[:2] 
    y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1) 
    fx,fy = flow[y,x].T 
    # create line endpoints 
    vector_scale = 3.0
    lines = np.vstack([x,y,x+fx*vector_scale,y+fy*vector_scale]).T.reshape(-1,2,2) 
    lines = np.int32(lines) 
    # create image and draw 
    vis = cv.cvtColor(im,cv.COLOR_GRAY2BGR) 
    for (x1,y1),(x2,y2) in lines: 
        cv.line(vis,(x1,y1),(x2,y2),(0,0,255),1) 
        cv.circle(vis,(x1,y1),1,(0,0,255), -1) 
    return vis

def draw_hsv_flow(flow, scale=10, sat=(2**8)-1):
	U, V = flow.shape[:2]
	# convert from cartesian to polar
	mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
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
	BGR = cv.cvtColor(HSV, cv.COLOR_HSV2BGR)
	return BGR

def load_data_set(path ='path_to_your_image_set'):
	images_list = []
	for i in range(7, 14+1):
		# 'frame' + "%02d" % (i,) + '.png'
		images_list.append(path +'frame' + "%02d" % (i,) + '.png')
		# images_list.append('./img/Yosemite/'+'frame' + "%02d" % (i,) + '.png')
	print (images_list[0])
	return images_list

# file_names = os.listdir('./img/Urban')
# for file_name in file_names:
# 	img_t = cv2.imread(file_name)
# 	print file_name+1


images_list = load_data_set('./img/Urban/')

for file in range(len(images_list)-1):
	"""
	Load images from dataset
	"""
	img_prior = cv.imread(images_list[file])
	img_current = cv.imread(images_list[file+1])
	"""
	Color conversion
	"""
	img_prior_ = cv.cvtColor(img_prior,cv.COLOR_BGR2GRAY)
	img_current_ = cv.cvtColor(img_current,cv.COLOR_BGR2GRAY)
	"""
	Gaussian smooth
	"""
	img_prior_ = cv.GaussianBlur(img_prior_,(3,3),1.5)
	img_current_ = cv.GaussianBlur(img_current_,(3,3),1.5)
	"""
	Farneback
	"""
	# grabs the initial time
	t0 = time.time()
	flow = cv.calcOpticalFlowFarneback(img_prior_, img_current_, flow=None,pyr_scale=0.5,levels=1,winsize=10,iterations=2,poly_n=7,poly_sigma=1.5,flags=0)
	# flow = cv2.medianBlur(flow, 5)
	# takes the final time
	t1 = time.time()
	print ("OF calculation took {:5f} ms" .format((t1-t0)*1000))

	cv.imshow('flow_field', draw_flow(img_prior_, flow,step=10))
	cv.imshow('flow_HSV', draw_hsv_flow(flow, scale=10, sat=(2**8)-1))
	cv.imshow('I(x,y,t+1)', img_prior)
	cv.waitKey(0)

# print img_prior.shape
# cv.imshow('t_', img_prior)
# cv.waitKey(0)
cv.destroyAllWindows()