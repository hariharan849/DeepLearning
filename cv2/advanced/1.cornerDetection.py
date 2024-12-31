""" Corner Detection
"""
from google.colab.patches import cv2_imshow
import cv2 
import numpy as np  

flat_chess = cv2.imread('sample_data/real_chessboard.jpg')
gray_flat_chess = cv2.cvtColor(flat_chess,cv2.COLOR_BGR2GRAY)


# Harris Corner Detection

# src Input
# dst Image
# blockSize Neighborhood size (see the details on #cornerEigenValsAndVecs ).
# ksize Aperture parameter for the Sobel operator.
# k Harris detector free parameter. See the formula in DocString

# Convert Gray Scale Image to Float Values
gray = np.float32(gray_flat_chess)

# Corner Harris Detection
dst = cv2.cornerHarris(gray,blockSize=2,ksize=3,k=0.04)

# result is dilated for marking the corners, not important to actual corner detection
# this is just so we can plot out the points on the image shown
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
flat_chess[dst>0.01*dst.max()]=[255,0,0]

cv2_imshow(flat_chess)





# Shi-Tomasi Corner Detector 

# image
# maxCorners Maximum number of corners to return.
# qualityLevel
# minDistance

corners = cv2.goodFeaturesToTrack(gray_flat_chess, 5, 0.01, 10)
# int0 is used to convert float values to integer
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(flat_chess, (x, y), 3, 255, -1)
    
cv2_imshow(flat_chess)