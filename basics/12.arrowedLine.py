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

#arrowedLine
cv2.arrowedLine(image, (50, 50), (200, 50), blue, 3, 8, 0, 0.1)
cv2.arrowedLine(image, (50, 120), (200, 120), green, 3, cv2.LINE_AA, 0, 0.3)

cv2_imshow(image)