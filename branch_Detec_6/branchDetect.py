import cv2
import numpy as np

# Read the binary image
image = cv2.imread('gdg.png', cv2.IMREAD_GRAYSCALE)

# Threshold the image to get binary image
_, image1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# from file 5 manuel binarized default image
image2 = cv2.imread('bdg.png')


# Define a kernel for dilation
kernel = np.ones((5,5), np.uint8)  # You can adjust the size of the kernel according to your requirement

# Dilate the image
image2 = cv2.dilate(image2, kernel, iterations=1)

# Convert image1 to color (3-channel) image
image1_color = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)

# Ensure both images have the same dimensions
image2 = cv2.resize(image2, (image1_color.shape[1], image1_color.shape[0]))

# Convert images to numpy arrays
image1_np = np.array(image1_color, dtype=np.int16)  # Use image1_color instead of image1
image2_np = np.array(image2, dtype=np.int16)

# Subtract one image from the other
result = image1_np - image2_np

# Clip negative values to zero
result = np.clip(result, 0, 255)

# Convert back to uint8
result = result.astype(np.uint8)

kernel = np.ones((3, 3), np.uint8)

# Perform erosion followed by dilation
result = cv2.erode(result, kernel, iterations=1)

# Display the result
cv2.imshow('Subtracted Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()