import cv2

"""
important notice : binarization of colored guess gives cells directy

Plan:

1- binarize new processed guess
2- binarized not processed guess
3- combine both binarized to fill in blanks 

"""

def binarize_image(image_path, threshold_value):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply binary thresholding
    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    
    return binary_image

threshold_value = 127  # Adjust this threshold value as needed

# new guess
image_path = 'guessedDenoisedGaussed.png'
binary_image = binarize_image(image_path, threshold_value)
# old guess
image_path = 'prediction_result1.jpg'
binary_image2 = binarize_image(image_path, threshold_value)


# Save the binary image
cv2.imwrite('binary_image.jpg', binary_image)
cv2.imwrite('binary_image2.jpg', binary_image2)
