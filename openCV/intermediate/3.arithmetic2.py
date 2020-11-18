"""
Arithmetic with images
"""

# Import required packages:
from google.colab.patches import cv2_imshow
import numpy as np
import cv2


# Load the original image:
image_1 = cv2.imread('sample_data/beagle.png')
image_2 = cv2.imread('sample_data/dogs_00036.jpg')
image_2 = cv2.resize(image_2, (image_2.shape[1], image_1.shape), interpolation=cv2.INTER_AREA)

print (image_1.shape, image_2.shape)
# alpha.img1+beta.img2+gamma
added_image = cv2.addWeighted(image_1, 0.3, image_2, 0.5, 0)


cv2_imshow(added_image)
