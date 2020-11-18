"""
Arithmetic with images
"""

# Import required packages:
from google.colab.patches import cv2_imshow
import numpy as np
import cv2

# Load the original image:
image = cv2.imread('sample_data/beagle.png')

# It is normally performed on binary images.



kernel = np.ones((5,5), np.uint8)
kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
"""
array([[1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]], dtype=uint8)
"""
# Elliptical Kernel
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
"""
array([[0, 0, 1, 0, 0],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [0, 0, 1, 0, 0]], dtype=uint8)
"""
# Cross-shaped Kernel
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
"""
array([[0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [1, 1, 1, 1, 1],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0]], dtype=uint8)
"""


"""
1. Erosion
The kernel slides through the image (as in 2D convolution).
A pixel in the original image (either 1 or 0) will be considered
 1 only if all the pixels under the kernel is 1,
otherwise it is eroded (made to zero).
"""
erosion = cv2.erode(image, kernel, iterations=1)

"""
2. Dilation(opposite of erosion)
"""
dilation = cv2.dilate(image, kernel, iterations=1)

"""
3. Gradient( difference between dilation and erosion )
"""
gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

"""
4. Opening(erosion followed by dilation.)
"""
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

"""
5. Closing(dilation followed by erosion.)
"""
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


cv2_imshow(image)
cv2_imshow(dilation)
