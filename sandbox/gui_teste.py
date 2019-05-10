import numpy as np
import cv2
import argparse

i = 0
# mouse callback function
def draw_ROI(event,x,y,flags,param):
	'''
	x-r,y-r   ------
	|          		|
	|          		|
	|          		|
	-------- x+r,y+r
	'''
	global i
	r = 20
	path = 'bge_dot_db/'
	
	colors = dict({
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0)})
	thickness = 2
	
	if event == cv2.EVENT_LBUTTONDBLCLK:
		i = i+1
		cv2.imwrite(path +'bge_dot_test_%02d.png' % i,img[y-r:y+r,x-r:x+r])
		# cv2.imwrite(path +'bge_dot_test_' + "%02d" + i + '.png',img[y-r:y+r,x-r:x+r])
		cv2.rectangle(img, (x-r, y-r), (x+r, y+r),colors['red'],thickness)
		
		print "bge_dot_test_ %2d " % i
		
        #cv2.circle(img,(x,y),20,(0,0,255),1)
        

# Create a black image, a window and bind the function to window
# img = np.zeros((512,512,3), np.uint8)
# load our serialized model from disk
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
print("[INFO] loading image...")
img = cv2.imread(args["image"])
height, width = img.shape[:2]
# print("[INFO] image size: %d" img.shape[:2])
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_ROI)

while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()