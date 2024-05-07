# importing libraries 
import numpy as np 
import cv2 
from matplotlib import pyplot as plt 


# Reading image from folder where it is stored 
img = cv2.imread('test.png') 


# denoising of image saving it into dst image 
dst = cv2.fastNlMeansDenoisingColored(img, None, 30, 10, 7, 5) 

"""
First parameter: This is the input image that you want to denoise. It should be a colored image in
 BGR format (OpenCV's default color format).

second parameter: This parameter is the output denoised image. Since it's set to None, the function will return the
 denoised image as the result.

third parameter : This is the strength of the denoising. A higher value means stronger denoising.

fourth parameter: This parameter controls the filter strength for luminance component. 
It's similar to the previous parameter but specifically for the luminance component (brightness).


fifth parameter: This is the size of the window for the noise estimation. 
A larger value means that more pixels will be considered when estimating the noise.

sixsth parameter : This is the size of the window for filtering. A larger value means that more pixels will 
be considered when filtering, which can result in smoother output but might lose some details.
"""


# Plotting of source and destination image 
plt.subplot(121), plt.imshow(img) 
plt.subplot(122), plt.imshow(dst) 

plt.show() 