""" Pyramid
"""
from google.colab.patches import cv2_imshow
import cv2 
import numpy as np  

dog_image = cv2.imread('sample_data/real_chessboard.jpg')
lower_reso = cv2.pyrDown(dog_image)
upper_reso = cv2.pyrUp(lower_reso)

cv2_imshow(dog_image)
cv2_imshow(lower_reso)
cv2_imshow(upper_reso)