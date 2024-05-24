"""
use array math to fill blanks
"""


import cv2

# Load the image
image_path = 'blended_image.jpg'
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is not None:
    # Display the shape of the image array (height, width, channels)
    print("Shape of the image array:", image.shape)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("", gray_image.shape)
print(gray_image[0,0])

# quşckly binarize image 
for i in range(0, 256):
    for j in range(0, 256):
        gray_image[i,j] = 0 if gray_image[i,j]<100 else 255

cv2.imwrite("manuelBinarized.png",gray_image)