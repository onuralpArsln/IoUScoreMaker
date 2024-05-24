import cv2
import numpy as np
from skimage.color import rgb2gray

# Load the image
img = cv2.imread('test.png')

# Split the image into its color channels
b, g, r = cv2.split(img)

# Create a new image with only the blue and red channels
new_img = cv2.merge((b, np.zeros_like(g), r))




# Display or save the result
cv2.imshow('Image without green channel', new_img)
cv2.imwrite('./channelClear.png', new_img) 


image_gray = rgb2gray(new_img)
blur = cv2.GaussianBlur(image_gray,(3,3),0)

cv2.imshow('Image w/o green and lp', blur)
cv2.imwrite('./channelClearFiltered.png', blur) 


ret,thresh1 = cv2.threshold(blur,0,150,cv2.THRESH_BINARY)

cv2.imshow('Image tresholded', thresh1)
cv2.imwrite('./threshold.png', thresh1) 

cv2.waitKey(0)
cv2.destroyAllWindows()

# If you want to save the result
#cv2.imwrite('output_image.jpg', new_img)