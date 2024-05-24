import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the binary image
image = cv2.imread('gdg.png', cv2.IMREAD_GRAYSCALE)

# Ensure the image is binary
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Define a kernel for erosion
kernel = np.ones((3, 3), np.uint8)

# Apply erosion multiple times
num_iterations = 1  # Adjust the number of iterations as needed
thin_image = binary_image.copy()
for i in range(num_iterations):
    thin_image = cv2.erode(thin_image, kernel, iterations=1)

# Display the original and thinned images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Binary Image')
plt.imshow(binary_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Thinned Image')
plt.imshow(thin_image, cmap='gray')
plt.axis('off')

plt.show()
