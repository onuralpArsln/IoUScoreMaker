import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize

# Load the binary image
image = cv2.imread('gdg.png', cv2.IMREAD_GRAYSCALE)

# Ensure the image is binary
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Invert the binary image for skeletonization
binary_image = cv2.bitwise_not(binary_image)

# Perform skeletonization (thinning)
thinned_image = skeletonize(binary_image)

# Invert back the thinned image
thinned_image = cv2.bitwise_not(thinned_image.astype(np.uint8))

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
