"""
OpenCV read tutorial
"""
from google.colab.patches import cv2_imshow
import numpy as np
import cv2

# Use the function cv2.imread() to read an image.
# The image should be in the working directory or a full path of image should be provided.
# load OpenCV logo image:
# Params: (image, flag)
#    image(str): Path of the image
#    flag(int): 0(cv2.IMREAD_GRAYSCALE), 1(cv2.IMREAD_COLOR) or -1(cv2.IMREAD_UNCHANGED)
image = cv2.imread("logo.png")

# split using numpy
(b, g, r) = image[:, :, 0], image[:, :, 1], image[:, :, 2]

# merging the channels
image = cv2.merge((r, b, r))

#First argument is the file name, second argument is the image you want to save.
cv2.imwrite("logo1.png", image)