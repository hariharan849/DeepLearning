"""
Thresholding with images
"""

# Import required packages:
from google.colab.patches import cv2_imshow
import numpy as np
import cv2


# Load the original image:
image_1 = cv2.imread('sample_data/beagle.png')

image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)

# Apply threshold

# 1. THRESH_BINARY
ret, thresh_binary = cv2.threshold(image_1, 127, 255, cv2.THRESH_BINARY)

# 2. THRESH_BINARY_INV
ret, thresh_binary_inv = cv2.threshold(image_1, 127, 255, cv2.THRESH_BINARY_INV)

# 3. cv2.THRESH_TRUNC
ret, thresh_trunc = cv2.threshold(image_1, 127, 255, cv2.THRESH_TRUNC)

# 4. cv2.THRESH_TOZERO
ret, thresh_to_zero = cv2.threshold(image_1, 127, 255, cv2.THRESH_TOZERO)

# 5. cv2.THRESH_TOZERO_INV
ret, thresh_to_zero_inv = cv2.threshold(image_1, 127, 255, cv2.THRESH_TOZERO_INV)


# OTSU(bimodel image(2 peaks))
ret, thresh_binary_otsu = cv2.threshold(image_1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# cv2.ADAPTIVE_THRESH_MEAN_C : threshold value is the mean of neighbourhood area.
# cv2.ADAPTIVE_THRESH_GAUSSIAN_C : threshold value is the weighted sum of neighbourhood values where weights are a gaussian window.
# Block Size - It decides the size of neighbourhood area.
# C - It is just a constant which is subtracted from the mean or weighted mean calculated.

th2 = cv2.adaptiveThreshold(image_1, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(image_1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2)

cv2_imshow(thresh_binary)
cv2_imshow(th3)
