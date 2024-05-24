# manually binarization of colored guess image  results the 
# binarized gues is  output of this file 

import cv2


def binarize_image(image_path, threshold_value):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply binary thresholding
    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    
    return binary_image

threshold_value = 127  # Adjust this threshold value as needed

image_path = 'prediction_result1.jpg'
if __name__ == "__main__":
    binary_image = binarize_image(image_path, threshold_value)
    cv2.imwrite('binarizedDefaultGuess.jpg', binary_image)

