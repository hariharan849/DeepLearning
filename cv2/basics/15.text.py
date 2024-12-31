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
image.fill(255)

# putText(img, text, pos, fontFace, fontScale, color, thickness, lineType)
cv2.putText(image, 'OpenCV4 with Python', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, red, 2,
            cv2.LINE_4)
cv2.putText(image, 'OpenCV4 with Python', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, green, 2,
            cv2.LINE_8)
cv2.putText(image, 'OpenCV4 with Python', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, blue, 2,
            cv2.LINE_AA)

cv2_imshow(image)
