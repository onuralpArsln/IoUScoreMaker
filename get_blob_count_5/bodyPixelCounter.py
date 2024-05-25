"""
aim is to open binarized image as array then count amount of pixels at each body 
blobs_detected is result of this
"""

import cv2
import numpy as np

# Load the binary image
image_path = 'binarizedDefaultGuess.jpg'
binary_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Ensure the image is binary
_, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)

# Find connected components (blobs)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

# Output results
for i in range(1, num_labels):  # Skip the background label 0
    x, y, w, h, area = stats[i]
    cx, cy = centroids[i]
    print(f"Cell Body {i}: Position (Centroid) = ({cx:.2f}, {cy:.2f}), Size (Area) = {area}")

    # Optionally, draw rectangles and centroids on the image for visualization
    cv2.rectangle(binary_image, (x, y), (x + w, y + h), (127), 2)
    cv2.circle(binary_image, (int(cx), int(cy)), 5, (127), -1)

# Save the result image with drawn blobs
cv2.imwrite('cells_detected.png', binary_image)