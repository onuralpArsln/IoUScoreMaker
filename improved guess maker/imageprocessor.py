# this image process make brightnes 3.4 and contrast 0.4 to improve readibility of image 


import cv2
import numpy as np


alpha = 0.4  # Contrast control (1.0 means no change)
beta = 3.4    # Brightness control (0 means no change)

input_image_path = 'C:\\Users\\onura\\Documents\\VSproject\linmig\\002.jpg'
input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)



def imgProces( img : np.ndarray) -> np.ndarray:
    adjusted_image = cv2.convertScaleAbs(input_image, alpha=alpha, beta=beta)
    
  

