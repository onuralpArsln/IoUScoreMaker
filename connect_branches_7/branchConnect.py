import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the binary image
image = cv2.imread('binary_image.png', cv2.IMREAD_GRAYSCALE)

# Ensure the image is binary
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Define a kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

# Perform morphological closing
closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# Display the original and closed images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Binary Image')
plt.imshow(binary_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Closed Image')
plt.imshow(closed_image, cmap='gray')
plt.axis('off')

plt.show()
