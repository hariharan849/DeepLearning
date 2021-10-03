"""
Introduction to contours

Contours can be explained simply as a curve joining all the continuous points (along the boundary), having same color or intensity.
The contours are a useful tool for shape analysis and object detection and recognition.
"""

# Import required packages:
from google.colab.patches import cv2_imshow
import numpy as np
import cv2


def draw_contour_outline(img, cnts, color, thickness=1):
    """Draws contours outlines of each contour"""

    for cnt in cnts:
        cv2.drawContours(img, [cnt], 0, color, thickness)


def build_sample_image():
    """Builds a sample image with basic shapes"""

    # Create a 500x500 gray image (70 intensity) with a rectangle and a circle inside:
    img = np.ones((500, 500, 3), dtype="uint8") * 70
    cv2.rectangle(img, (100, 100), (300, 300), (255, 0, 255), -1)
    cv2.circle(img, (400, 400), 100, (255, 255, 0), -1)

    return img



# Load the image and convert it to grayscale:
image = build_sample_image()
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply cv2.threshold() to get a binary image:
ret, thresh = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)


""" It stores the (x,y) coordinates of the boundary of a shape. But does it store all the coordinates ?
That is specified by this contour approximation method.

 cv2.CHAIN_APPROX_NONE, all the boundary points are stored.
 cv2.CHAIN_APPROX_SIMPLE removes all redundant points and compresses the contour, thereby saving memory.
"""
# Find contours using the thresholded image:
# Note: cv2.findContours() has been changed to return only the contours and the hierarchy
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours2, hierarchy2 = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours2, hierarchy2 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Show the number of detected contours for each call:
print("detected contours (RETR_EXTERNAL): '{}' ".format(len(contours)))
print("detected contours (RETR_LIST): '{}' ".format(len(contours2)))

# Copy image to show the results:
image_contours = image.copy()
image_contours_2 = image.copy()

for cnt in contours2:
        cv2.drawContours(image, [cnt], 0, (0, 0, 255), 1)

cv2_imshow(image)









""" Moments
    
    Mass, centroid and area of the object
"""
cnt = contours[0]
M = cv2.moments(cnt)
print (M)

#centriod
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

#contour area
area = cv2.contourArea(cnt)

#perimeter
perimeter = cv2.arcLength(cnt,True)