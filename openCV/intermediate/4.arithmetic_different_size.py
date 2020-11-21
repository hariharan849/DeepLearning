"""
Sobel operator example in order to see how this operator works and how cv2.addWeighted() can be used
"""

# Import required packages:
import cv2
from google.colab.patches import cv2_imshow

# Load the original image:
image_1 = cv2.imread('sample_data/beagle.png')
image_2 = cv2.imread('sample_data/logo.png')

rows, cols, channels = image_2.shape

#Extract ROI
roi = image_1[0:rows, 0:cols]

#convert to grayscale
image_2_gray = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

# Apply threshold
ret, mask = cv2.threshold(image_2_gray, 10, 255, cv2.THRESH_BINARY)

# inverse mask
mask_inv = cv2.bitwise_not(mask)

#form bg and fg
image_1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
image_2_bg = cv2.bitwise_and(image_2, image_2, mask=mask)

dst = cv2.add(image_1_bg, image_2_bg)
image_1[0:rows, 0:cols] = dst

cv2_imshow(image_1)