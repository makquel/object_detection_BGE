import cv2
# Print version string
print "OpenCV version :  {0}".format(cv2.__version__)
 
# Extract major, minor, and subminor version numbers 
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print "Major version :  {0}".format(major_ver)
print "Minor version :  {0}".format(minor_ver)
print "Submior version :  {0}".format(subminor_ver)
 
if int(major_ver) < 3 :
    '''
        Old OpenCV 2 code goes here
    '''
else :
    '''
        New OpenCV 3 code goes here 
    '''