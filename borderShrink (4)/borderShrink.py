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
image_path2 = 'prediction_result1.jpg'
binary_image2 = binarize_image(image_path2, threshold_value)


# Save the binary images
cv2.imwrite('binary_image.jpg', binary_image)
cv2.imwrite('binary_image2.jpg', binary_image2)


# combine images

def combine_images(image1_path, image2_path, output_path, alpha=0.5):
    # Load images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    
    # Resize images to have the same dimensions (optional)
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    
    # Blend images
    blended_image = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)
    
    # Save the blended image
    cv2.imwrite(output_path, blended_image)

# Example usage

output_path = 'blended_image.jpg'
combine_images('binary_image.jpg', 'binary_image2.jpg', output_path, alpha=0.5)