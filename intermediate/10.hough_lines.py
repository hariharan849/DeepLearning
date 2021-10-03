import cv2
from google.colab.patches import cv2_imshow
import numpy as np

img = cv2.imread('lines.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 120)
"""
 Probabilistic hough Lines
 Parameters:
 image
 rho - positional step size in pixels
 theta - rotational step size in radians.
 (Search for lines seperated by 1 pixel and 1 degree)
 threshold
 minLineLength
 maxLineGap
"""
minLineLength = 20
maxLineGap = 5
lines = cv2.HoughLinesP(edges, 1, np.pi/180.0, 20,
                        minLineLength, maxLineGap)
for x1, y1, x2, y2 in lines[0]:
    cv2.line(img, (x1, y1), (x2, y2), (0,255,0),2)

cv2_imshow(edges)
cv2_imshow(img)
cv2.destroyAllWindows()
