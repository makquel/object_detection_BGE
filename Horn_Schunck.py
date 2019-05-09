from scipy.ndimage import imread
from scipy import signal
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import convolve as filter2
import matplotlib.pyplot as plt
# import cv2

print("This code executes Horn - Schunck algorithm") 

"""
the assumption is that groups of pixels move similarly, but not the same.
"""
"""
https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
"""
def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

# im1_ = imread("./img/synth_img_1.png",flatten=True).astype(float)
# im2_ = imread("./img/synth_img_2.png",flatten=True).astype(float)
im1_ = imread("./img/Urban/frame07.png",flatten=True).astype(float)
im2_ = imread("./img/Urban/frame08.png",flatten=True).astype(float)
# im1_ = imread("./img/darcy_3.png",flatten=True).astype(float)
# im2_ = imread("./img/darcy_4.png",flatten=True).astype(float)

ly, lx = im1_.shape
print("lx", lx, "ly", ly)
# im1_= crop_center(im1_,430,400)
# im2_= crop_center(im2_,430,400)


# crop_face = im1_[181:377,450:717] 
# print("size", im1_.shape)
# plt.imshow(crop_face)
# plt.show()
"""
Gaussian filter for non synthetic images
"""
sigma = 2
order = 0
im1_ = gaussian_filter(im1_,sigma, order) #
im2_ = gaussian_filter(im2_,sigma, order) #

"""
Gradient (first-order derivatives)
"""
dx_kernel = (1./4)*np.array([[-1, 1],[-1, 1]])
dy_kernel = (1./4)*np.array([[-1, -1],[1, 1]])
dt_kernel = (1./4)*np.ones([2,2])

# fx = signal.convolve2d(im1_, dx_kernel, boundary='symm','same') + signal.convolve2d(im2_, dx_kernel, boundary='symm','same')
fx = filter2(im1_,dx_kernel) + filter2(im2_,dx_kernel)
fy = filter2(im1_,dy_kernel) + filter2(im2_,dy_kernel)
ft = filter2(im1_,dt_kernel) + filter2(im2_,-dt_kernel)
"""
Algorithm

Alpha affects the weight of the gradient in the calculation of the motion vector.
The bigger lambda is, the smaller the effect of the gradient in the u and v vectors, and the larger the effect of the Laplacian
Greater lambda produces greater directional accuracy in the vectors
Greater lambda produces shorter direction vectors, distorting the displacement magnitudes slightly.
The number of iterations affects the sensitivity and accuracy of the detection.
"""
# print "Image t_0 size: ",im1_.shape
u = np.zeros([im1_.shape[0],im1_.shape[1]])
v = np.zeros([im1_.shape[0],im1_.shape[1]])
ite = 100; #number of iterations
alpha = 10; #smoothness coefficient
fig,ax = plt.subplots(1,3,figsize=(18,5))
for f,a,t in zip((fx,fy,ft),ax,('$f_x$','$f_y$','$f_t$')):
	h = a.imshow(f,cmap='bwr')
    # a.set_title(t)
	fig.colorbar(h,ax=a)
# plt.show()
kernel_l = np.array([[1./12, 1./6, 1./12],[1./6, 0., 1./6],[1./12, 1./6, 1./12]])

# print(fx[100,100],fy[100,100],ft[100,100])

alpha = 1
ite = 20



for it in range(ite):
	u_avg = filter2(u,kernel_l)
	# u_avg = signal.convolve2d(u, kernel_l, boundary='symm',mode='same') #this method takes more time though 
	v_avg = filter2(v,kernel_l)
	gamma = (fx*u_avg + fy*v_avg + ft) / (alpha**2 + fx**2 + fy**2)
	u = u_avg - fx * gamma
	v = v_avg - fy * gamma

# if __name__ == "__main__":
"""
Plot Results
"""   
ax = plt.figure().gca()
ax.imshow(im2_,cmap = 'gray')
# plt.scatter(POI[:,0,1],POI[:,0,0])
scale = 3
# print len(v)
step_size = 10
for i in range(0,len(u),step_size):
	for j in range(0,len(v),step_size):
		ax.arrow(j,i, v[i,j]*scale, u[i,j]*scale, color='red')

	# plt.arrow(POI[:,0,0],POI[:,0,1],0,-5)

# plt.draw(); 
# plt.pause(0.01)
plt.show()