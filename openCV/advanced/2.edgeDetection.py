""" Corner Detection
"""
from google.colab.patches import cv2_imshow
import cv2 
import numpy as np  

img = cv2.imread('sample_data/sammy_face.jpg')
edges = cv2.Canny(image=img, threshold1=127, threshold2=127)
cv2_imshow(edges)



# Calculate the median pixel value
med_val = np.median(img) 
# Lower bound is either 0 or 70% of the median value, whicever is higher
lower = int(max(0, 0.7* med_val))
# Upper bound is either 255 or 30% above the median value, whichever is lower
upper = int(min(255,1.3 * med_val))
edges = cv2.Canny(image=img, threshold1=lower , threshold2=upper)
cv2_imshow(edges)



blurred_img = cv2.blur(img,ksize=(5,5))
edges = cv2.Canny(image=blurred_img, threshold1=lower , threshold2=upper)
cv2_imshow(edges)