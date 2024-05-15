"""
aim is to open binarized image as array then count amount of pixels at each body 
"""

import cv2


image=cv2.imread("binarizedDefaultGuess.jpg")
image_matrix = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


