"""
Example to show how to draw line using OpenCV
"""
from google.colab.patches import cv2_imshow
import cv2
import numpy as np

#colors
light_gray = (220, 220, 220)
green = (0, 255, 0)
blue = (255, 0, 0)
red = (0, 0, 255)
magenta = (255, 0, 255)
cyan = (255, 255, 0)

# We create the canvas to draw: 400 x 400 pixels, 3 channels, uint8 (8-bit unsigned integers)
# We set the background to black using np.zeros():
image = np.zeros((400, 400, 3), dtype="uint8")
image[:] = light_gray

# We are going to draw several polylines
# These points define a triangle:
pts = np.array([[250, 5], [220, 80], [280, 80]], np.int32)
# Reshape to shape (number_vertex, 1, 2)
pts = pts.reshape((-1, 1, 2))
# Print the shapes: this line is not necessary, only for visualization:
print("shape of pts '{}'".format(pts.shape))

# Draw this polygon with True option:
cv2.polylines(image, [pts], True, green, 3)
cv2.fillPoly(image, [pts], green)

cv2_imshow(image)
