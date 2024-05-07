import cv2
import numpy as np

# Load the image
img = cv2.imread('test.png')

# Split the image into its color channels
b, g, r = cv2.split(img)

# Create a new image with only the blue and red channels
new_img = cv2.merge((b, np.zeros_like(g), r))

# Display or save the result
cv2.imshow('Image without green channel', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# If you want to save the result
#cv2.imwrite('output_image.jpg', new_img)