import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.util import invert
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('gdg.png', cv2.IMREAD_GRAYSCALE)

# Step 1: Preprocess the Image
# Apply GaussianBlur to reduce noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Step 2: Binary Thresholding
# Invert the image since the cells/branches are white
image_inverted = invert(blurred)

# Use Otsu's thresholding on the inverted image
_, binary = cv2.threshold(image_inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Step 3: Skeletonization
# Perform skeletonization
skeleton = skeletonize(binary // 255)

# Convert skeleton to uint8 format for OpenCV
skeleton = (skeleton * 255).astype(np.uint8)

# Step 4: Detect and Trace the Skeleton
# Find contours of the skeleton
contours, _ = cv2.findContours(skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create an output image to draw the contours
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Draw contours (branches) on the output image
cv2.drawContours(output_image, contours, -1, (0, 255, 0), 1)

# Display the results
plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 3, 2)
plt.title('Skeleton')
plt.imshow(skeleton, cmap='gray')
plt.subplot(1, 3, 3)
plt.title('Traced Branches')
plt.imshow(output_image)
plt.show()
