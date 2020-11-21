"""
Arithmetic with images
"""

# Import required packages:
from google.colab.patches import cv2_imshow
import numpy as np
import cv2


# Load the original image:
image = cv2.imread('sample_data/sudoku.png')

# laplacian derivatives
laplacian = cv2.Laplacian(image, cv2.CV_64F)
# joint Gausssian smoothing plus differentiation operation
# resistent to noise
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

cv2_imshow(image_1)
cv2_imshow(laplacian)
cv2_imshow(sobelx)
cv2_imshow(sobely)

