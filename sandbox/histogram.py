import cv2
import numpy as np 
from matplotlib import pyplot as plt

img = cv2.imread('./teste_1.png')
# img = cv2.imread('./darcy_surgery_basis.png')


color = ('b', 'g', 'r')
u,v,nchan = img.shape
img = img[100:400,300:600]
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])

bg_color = 'black'
fg_color = 'white'

fig = plt.figure(facecolor=bg_color, edgecolor=fg_color)
axes = fig.add_subplot(111)
axes.patch.set_facecolor(bg_color)
axes.xaxis.set_tick_params(color=fg_color, labelcolor=fg_color)
axes.yaxis.set_tick_params(color=fg_color, labelcolor=fg_color)
for spine in axes.spines.values():
    spine.set_color(fg_color)
plt.grid(True)
plt.plot(hist, color='red')
plt.xlim([0, 256])
plt.xlabel('$bins$', color=fg_color)
plt.ylabel('$pixels$', color=fg_color)
plt.show()

img_eq = cv2.equalizeHist(img_gray)
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    # plt.plot(histr, color = col)
    plt.xlim([0, 256])
    plt.ylim([0,(u*v)/100])
# plt.show()

# plt.grid(True)
# plt.hist(img_gray.ravel(),256,[0,256],facecolor='b') 
# plt.hist(img_eq.ravel(),256,[0,256],facecolor='g',alpha=0.75)
# plt.hist(img_eq.ravel(),256,[0,256],facecolor='r',alpha=0.75)
# plt.show()

cv2.imshow("Original/Equalizada", np.hstack([img_gray, img_eq]))

cv2.waitKey(0)

cv2.destroyAllWindows()