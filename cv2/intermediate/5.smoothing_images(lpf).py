"""
Arithmetic with images
"""

# Import required packages:
from google.colab.patches import cv2_imshow
import numpy as np
import cv2


# Load the original image:
image_1 = cv2.imread('sample_data/beagle.png')

"""
LPF(low pass filter): removing noise, blurring
HPF(high pass filter): finding edges
"""
kernel = np.ones((5, 5), np.float32)/25
# -1 -> ddepth
dst = cv2.filter2D(image_1, -1, kernel)

# Above doesnt care about edges
# to preserve edges
blur = cv2.blur(image_1,(5,5))

# gaussian filter
blur = cv2.GaussianBlur(image_1, (5, 5), 1)

# median filter(
# computes the median of all the pixels under the kernel window and
# the central pixel is replaced with this median value.
# This is highly effective in removing salt-and-pepper noise)
blur = cv2.medianBlur(image_1,5)

cv2_imshow(image_1)
cv2_imshow(blur)
