import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the binary image
image = cv2.imread('guessedDenoisedGaussed.png', cv2.IMREAD_GRAYSCALE)

# Ensure the image is binary
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Apply thinning
thinned_image = cv2.ximgproc.thinning(binary_image)

# Display the original and thinned images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Binary Image')
plt.imshow(binary_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Thinned Image')
plt.imshow(thinned_image, cmap='gray')
plt.axis('off')

plt.show()
