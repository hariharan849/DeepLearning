"""
Arithmetic with images
"""

# Import required packages:
from google.colab.patches import cv2_imshow
import numpy as np
import cv2


# Load the original image:
image = cv2.imread('sample_data/beagle.png')

# Add 60 to every pixel on the image. The result will look lighter:
M = np.ones(image.shape, dtype="uint8") * 60
added_image = cv2.add(image, M)

# Subtract 60 from every pixel. The result will look darker:
subtracted_image = cv2.subtract(image, M)

cv2_imshow(M)
cv2_imshow(image)
cv2_imshow(added_image)
cv2_imshow(subtracted_image)