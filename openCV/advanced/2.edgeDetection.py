""" Corner Detection
"""
from google.colab.patches import cv2_imshow
import cv2 
import numpy as np  


"""
Theory
======

1. Noise Reduction
Since edge detection is susceptible to noise in the image,
first step is to remove the noise in the image with a 5x5 Gaussian
filter.

2.Finding Intensity Gradient of the Image
Smoothened image is then filtered with a Sobel kernel
in both horizontal and vertical direction to get first
derivative in horizontal direction (G_x) and
vertical direction (G_y).
From these two images, we can find edge gradient and
direction for each pixel as follows:

Edge_Gradient (G) = sqrt{G_x^2 + G_y^2}

Angle (theta) = tan^{-1}({G_y}/{G_x})

Gradient direction is always perpendicular to edges. It is rounded to one of four angles representing vertical, horizontal and two diagonal directions.

3. Non-maximum Suppression
After getting gradient magnitude and direction,
a full scan of image is done to remove any unwanted
pixels which may not constitute the edge.
For this, at every pixel, pixel is checked if it
is a local maximum in its neighborhood in the direction
 of gradient.


4. Hysteresis Thresholding
This stage decides which are all edges are really edges
and which are not. For this, we need two threshold
 values, minVal and maxVal. Any edges with intensity
 gradient more than maxVal are sure to be edges and
 those below minVal are sure to be non-edges, so
 discarded. Those who lie between these two thresholds
 are classified edges or non-edges based on their
 connectivity. If they are connected to “sure-edge”
 pixels, they are considered to be part of edges.
 Otherwise, they are also discarded. 
"""

""" Parameter
    minVal
    maxValue
    apertureSize(sobel kernel): default 3
    default: false(Edge_Gradient (G) = |G_x| + |G_y|)
"""
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