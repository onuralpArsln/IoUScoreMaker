import cv2
import numpy as np

# Load the image
img = cv2.imread('test.png')

# Split the image into its color channels
b, g, r = cv2.split(img)

# Create a new image with only the blue and red channels
new_img = cv2.merge((b, np.zeros_like(g), r))


image_gray = cv2.rgb2gray(new_img)
blur = cv2.GaussianBlur(image_gray,(3,3),0)

# Display or save the result
cv2.imshow('Image without green channel', new_img)
cv2.imwrite('./channelClear.png', new_img) 

cv2.imshow('Image w/o green and lp', blur)
cv2.imwrite('./channelClearFiltered.png', blur) 

cv2.waitKey(0)
cv2.destroyAllWindows()

# If you want to save the result
#cv2.imwrite('output_image.jpg', new_img)